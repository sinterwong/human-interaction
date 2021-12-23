import os
import numpy as np
from flask.helpers import send_file
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.secret_key = "s3cr3t"
app.debug = True
app._static_folder = os.path.abspath("templates/static/")


@app.route("/", methods=["GET"])
def index():
    title = "Gathering the face info"
    return render_template("layouts/index.html", title=title)


@app.route("/faces", methods=["POST"])
def post_faces():
    face_info = request.form["detections"]
    width = request.form["width"]
    height = request.form["height"]
    return jsonify(face_info)


@app.route("/faces_recognition", methods=["POST"])
def post_faces_recognition():
    result = eval(request.form["detections"])
    descriptor = result["descriptor"]
    features = np.array(list(descriptor.values())).astype(np.float32)

    return jsonify(result)


@app.route("/hands", methods=["POST"])
def post_hands_points():
    hands_points = request.form["pose_landmarks"]
    width = request.form["width"]
    height = request.form["height"]
    return jsonify(hands_points)


@app.route("/models/<path:name>", methods=["GET"])
def models(name):
    return send_file(os.path.join("models", name))


@app.route("/labeled_images/<path:name>/<path:idx>", methods=["GET"])
def face_images(name, idx):
    return send_file(os.path.join("labeled_images", name, idx))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
