from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
from roboflow import Roboflow 
import tempfile
import os

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler for /detect POST endpoint.
    Receives an uploaded file with the name "image_file", passes it
    through the YOLOv8 object detection network, and returns an array
    of bounding boxes.
    :return: JSON array of objects' bounding boxes in format [[x1, y1, x2, y2, object_type, probability],..]
    """
    # if 'image_file' not in request.files:
    #     return jsonify({"error": "No file part"}), 400

    file = request.files["image_file"]

    # if file.filename == '':
    #     return jsonify({"error": "No selected file"}), 400

    boxes = detect_objects_on_image(file.stream)
    return jsonify(boxes)


def detect_objects_on_image(file):
    """
    Receives an image, passes it through the YOLOv8 neural network,
    and returns an array of detected objects and their bounding boxes.
    :param file: Input image file stream
    :return: Array of bounding boxes in format [[x1, y1, x2, y2, object_type, probability],..]
    """
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
    #     # Save the file stream to a temporary file
    #     temp_file.write(file.read())
    #     temp_file_path = temp_file.name

    # try:
        # rf = Roboflow(api_key="7tmllls4BTvoWMy9lAy7")
        # project = rf.workspace().project("plants-images")
        # model = project.version(8).model
    model = YOLO("yolov8n.pt")
        # results = model.predict(temp_file_path)
    results = model.predict(Image.open(file))
    result = results[0]

    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
    # finally:
        # Clean up: remove the temporary file
        # os.remove(temp_file_path)

    return output


serve(app, host='0.0.0.0', port=8080)
