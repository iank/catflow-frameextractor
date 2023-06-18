import cv2
import numpy as np
from PIL import Image
from flask import current_app
from pgvector.psycopg2 import register_vector


def yolo_to_ls(x_px, y_px, width_px, height_px, original_width, original_height):
    """Convert YOLO bounding box to Label Studio format

    (x/y centered, units pixels) to (x/y top left, units proportion of original image)
    """
    # Scale
    x = x_px * 100.0 / original_width
    y = y_px * 100.0 / original_height
    width = width_px * 100.0 / original_width
    height = height_px * 100.0 / original_height

    # Convert (x, y) from center of bounding box to top left corner
    x = x - width / 2
    y = y - height / 2

    return x, y, width, height


def get_predictions(model, image_path):
    # Open the image file
    pil_image = Image.open(image_path)

    # Get original image width and height
    original_width, original_height = pil_image.size

    # Perform inference
    results = model.predict(pil_image)

    predictions = {"model_version": model.model_name, "score": 0.5, "result": []}

    # Transform result into required format
    for idx, pred in enumerate(results):
        if pred.confidence < model.threshold:
            continue

        x, y, width, height = yolo_to_ls(
            pred.x, pred.y, pred.width, pred.height, original_width, original_height
        )

        predictions["result"].append(
            {
                "id": f"result{idx+1}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [pred.label],
                },
            }
        )

    return predictions


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fc = 0
    ret = True

    interesting_frames = []

    while fc < frameCount and ret:
        ret, frame = cap.read()

        if not ret:
            continue

        interesting_frames.append(frame)
        fc += 1

    cap.release()
    return interesting_frames


def vector_db_add_novel(
    db, feature_extractor, file_path, threshold, frame_uuid, video_name
):
    register_vector(db)

    embedding = np.array(feature_extractor.get_vector(file_path).tolist())

    query = (
        "SELECT uuid, cosine_distance(embedding, %s) as distance"
        " FROM images ORDER BY distance LIMIT 1"
    )
    cur = db.cursor()
    cur.execute(query, (embedding,))
    results = cur.fetchall()
    cur.close()

    (nearest_uuid, min_distance) = results[0]  # first/only result
    current_app.logger.debug(
        f"cosine distance to {nearest_uuid} (nearest): {min_distance:.4f}"
    )
    if min_distance < threshold:
        return False

    # Novel image, add to DB
    cur = db.cursor()
    insertQuery = "INSERT INTO images (uuid, embedding, source) VALUES (%s, %s, %s)"
    cur.execute(insertQuery, (frame_uuid, embedding, video_name))
    db.commit()
    cur.close()

    return True
