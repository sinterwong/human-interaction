import os
import numpy as np
import pickle as pk
from flask.helpers import send_file
from flask import Flask, jsonify, render_template, request
from Features import FeaturesManager

app = Flask(__name__)
app.secret_key = "s3cr3t"
app.debug = True
app._static_folder = os.path.abspath("templates/static/")


@app.route("/", methods=["GET"])
def index():
    title = "Gathering the face info"
    return render_template("layouts/index.html", title=title)


@app.route("/initFeatureLibraries", methods=["POST"])
def post_init_feature_libraries():

    descriptions = eval(request.form["descriptions"])

    people_features = {}
    for p in descriptions:
        for name, s_feature in p:
            feature = s_feature.values()
            feature = np.array(list(feature), dtype=np.float32)
            people_features[name] = feature
    features = np.array(list(people_features.values()))
    insert_num = features_manager.mult_insert(
        features=features, names=list(people_features.keys()))

    with open(fm_path, 'wb') as wf:
        pk.dump(features_manager, wf)

    result = {
        "insert number": insert_num,
        "update number": len(people_features) - insert_num
    }
    return jsonify(result)


@app.route("/faces", methods=["POST"])
def post_faces():
    face_info = request.form["detections"]
    return jsonify(face_info)


@app.route("/faces_recognition", methods=["POST"])
def post_faces_recognition():
    detections = eval(request.form["detections"])

    results = []
    for det in detections:
        res = {}
        descriptor = det["descriptor"]
        feature = np.array(list(descriptor.values())).astype(np.float32)
        name, score = features_manager.identify(feature)
        res["name"] = name
        res["score"] = "%.3f" % score
        results.append(res)
    return jsonify(results)


@app.route("/hands", methods=["POST"])
def post_hands_points():
    hands_points = request.form["pose_landmarks"]
    width = request.form["width"]
    height = request.form["height"]
    return jsonify(hands_points)


@app.route("/faceMesh", methods=["POST"])
def post_fece_mesh_points():
    pose_landmarks = request.form["pose_landmarks"]
    width = request.form["width"]
    height = request.form["height"]
    result = {
        "status": True,
        "other": "hello"
    }
    return jsonify(result)


@app.route("/models/<path:name>", methods=["GET"])
def models(name):
    return send_file(os.path.join("models", name))


@app.route("/labeled_images/<path:name>/<path:idx>", methods=["GET"])
def face_images(name, idx):
    return send_file(os.path.join("labeled_images", name, idx))


if __name__ == "__main__":
    fm_path = "models/feature_libraries.pkl"
    if not os.path.exists(fm_path):
        features_manager = FeaturesManager()
    else:
        with open(fm_path, mode="rb") as rf:
            features_manager = pk.load(rf)
            features_manager.threshold = 0.938
    # face feature libraries
    app.run(host="127.0.0.1", port=5000, debug=True)
