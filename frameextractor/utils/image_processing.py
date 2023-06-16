import cv2
import numpy as np
from PIL import Image
from flask import current_app
from pgvector.psycopg2 import register_vector


def get_predictions(model, model_name, image_path, threshold):
    # Open the image file
    pil_image = Image.open(image_path)

    # Get original image width and height
    original_width, original_height = pil_image.size

    # Perform inference
    results = model(pil_image)

    # Get bounding boxes and labels
    boxes = results.xywh[0]
    labels = results.xywh[0][:, -1].int()

    predictions = {"model_version": model_name, "score": 0.5, "result": []}

    # Transform result into required format
    i = 0
    for box, label in zip(boxes, labels):
        x, y, width, height, confidence, class_id = box
        if confidence < threshold:
            continue

        x = x * 100.0 / original_width
        y = y * 100.0 / original_height
        width = width * 100.0 / original_width
        height = height * 100.0 / original_height

        # Convert (x, y) from center of bounding box to top left corner
        x = x - width / 2
        y = y - height / 2

        predictions["result"].append(
            {
                "id": f"result{i+1}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": x.item(),
                    "y": y.item(),
                    "width": width.item(),
                    "height": height.item(),
                    "rectanglelabels": [results.names[label.item()]],
                },
            }
        )
        i = i + 1

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
