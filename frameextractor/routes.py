# flaskapp/routes.py
from flask import request, abort, jsonify, current_app, Response
from .utils.s3_connector import S3Connector
from .utils.label_studio import LabelStudioAPI
from .utils.model import load_model
from .utils.db import vector_db_connect
from .utils.embedding import ImageFeatureExtractor
from .utils.image_processing import extract_frames, get_predictions, vector_db_add_novel
from flask_executor import Executor
import pkg_resources
import tempfile
import os
import time
import uuid
import cv2

def register_routes(app):
    executor = Executor(app)
    model = load_model(app.config['MODEL_NAME'])
    feature_extractor = ImageFeatureExtractor()


    @app.route('/status')
    def status():
        status_info = {
            "status": "OK",
            "service_name": "frameextractor",
            "version": pkg_resources.require("frameextractor")[0].version
        }

        return jsonify(status_info), 200


    @app.route('/sources', methods=['POST'])
    def sources():
        secret_token = app.config['SECRET_TOKEN']
        
        if 'Authorization' not in request.headers or request.headers['Authorization'] != secret_token:
            abort(401)  # Unauthorized

        uuids = request.json.get('uuids', [])
        if not isinstance(uuids, list) or not all(isinstance(i, str) for i in uuids):
            return jsonify({"error": "Invalid input. uuids should be a list of strings"}), 400
        sources = get_sources(uuids, app.config['PG_CONFIG'])
        return jsonify(sources)


    @app.route('/export', methods=['GET'])
    def export():
        api = LabelStudioAPI(app.config['LABELSTUDIO_BASE_URL'], app.config['LABELSTUDIO_AUTH_TOKEN'])

        try:
            # Create a snapshot on the server
            export_id, snapshot_name = api.create_snapshot(app.config['PROJECT_ID'])

            # Check the status until the snapshot is created
            completed = False
            while not completed:
                completed = api.check_export_status(app.config['PROJECT_ID'], export_id)
                time.sleep(1)

            # Convert the created snapshot to YOLO
            api.convert_snapshot(app.config['PROJECT_ID'], export_id, "YOLO")

            # Check the status until the conversion is complete
            completed = False
            while not completed:
                completed = api.check_conversion_status(app.config['PROJECT_ID'], export_id, "YOLO")
                time.sleep(1)

            # Download the archive
            resp = api.download_snapshot(app.config['PROJECT_ID'], export_id, snapshot_name)

            def generate():
                for chunk in resp.iter_content(chunk_size=1024):
                    yield chunk

            response = Response(generate(), content_type=resp.headers['content-type'])
            response.headers["Content-Disposition"] = f"attachment; filename={snapshot_name}.zip"

            return response

        except Exception as e:
            app.logger.error(f"/export: {e}")
            abort(500)

    
    @app.route('/process_video', methods=['POST'])
    def process_video():
        secret_token = app.config['SECRET_TOKEN']

        if 'Authorization' not in request.headers or request.headers['Authorization'] != secret_token:
            abort(401)  # Unauthorized

        # Check if the post request has the file part
        if 'file' not in request.files:
            abort(400)  # Bad Request

        file = request.files['file']
        if file.filename == '':
            abort(400)  # Bad Request

        # Save uploaded file to a temporary file
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, file.filename)
        file.save(temp_file)

        future = executor.submit(process_video_file, model, feature_extractor, app.config, temp_dir, temp_file)

        # Temporary debug
#        try:
#            future.result()
#        except Exception as e:
#            current_app.logger.error(f"Error processing video: {e}")
#            return jsonify({"message": f"Error processing video: {e}"}), 500

        return jsonify({"message": "Processing started"}), 202


def process_video_file(model, feature_extractor, config, temp_dir, temp_file):
    current_app.logger.info(f'Processing video file {temp_file}')

    # Save video name for the DB
    video_name = os.path.basename(temp_file)

    # Connect to S3
    connector = S3Connector(
            config['S3_CONFIG']['endpoint_url'],
            config['S3_CONFIG']['access_key_id'],
            config['S3_CONFIG']['secret_access_key']
    )
    # Connect to vector DB
    db = vector_db_connect(config['PG_CONFIG'])

    # Create Label Studio API client
    labelstudio = LabelStudioAPI(config['LABELSTUDIO_BASE_URL'], config['LABELSTUDIO_AUTH_TOKEN'])

    # Process video and save motion frames
    frames = extract_frames(temp_file)

    tasks_created = 0
    with tempfile.TemporaryDirectory() as frame_dir:
        for frame in frames:
            frame_uuid = str(uuid.uuid4())
            frame_filename = f"{frame_uuid}.png"

            temp_file_path = os.path.join(frame_dir, frame_filename)

            cv2.imwrite(temp_file_path, frame)

            # Check that the min distance to a frame already in our vector DB is > threshold
            if not vector_db_add_novel(db, feature_extractor, temp_file_path, config['VDB_THRESHOLD'], frame_uuid, video_name):
                continue

            # Get predictions
            predictions = get_predictions(model, config['MODEL_NAME'], temp_file_path, config['THRESHOLD'])

            # Upload to S3
            s3_uri = config['S3_CONFIG']['endpoint_url'] + '/' + connector.upload_file(config['S3_CONFIG']['bucket_name'], temp_file_path, frame_filename)

            if s3_uri is None:
                current_app.logger.error(f'Failed to upload {frame_uuid} to S3')

            current_app.logger.debug(f'S3 URI: {s3_uri}')

            # Create task
            task = {
                'data': {
                    'image': s3_uri
                },
                'predictions': [predictions]
            }

            try:
                response = labelstudio.import_task(config['PROJECT_ID'], task)
                if response is None:
                    current_app.logger.error(f'Failed to upload task to Label Studio')
                else:
                    current_app.logger.info(f'Created task {frame_uuid}')
                    tasks_created = tasks_created + 1
            except Exception as e:
                current_app.logger.error(f'Failed to upload task to Label Studio: {e}')

    # Log
    current_app.logger.info(f"Tasks created: {tasks_created}/{len(frames)}")

    # Cleanup
    temp_dir.cleanup()
    db.close()

    return '', 200  # OK


def get_sources(uuids, db_config):
    conn = vector_db_connect(db_config)
    cur = conn.cursor()
    query = "SELECT uuid, source FROM images WHERE uuid IN %s"
    cur.execute(query, (tuple(uuids),))
    rows = cur.fetchall()
    conn.close()
    return rows
