import io

import requests

from flask import Flask, jsonify, request

from werkzeug.middleware.proxy_fix import ProxyFix

from src.main_predictor import MainPredictor

app = Flask(__name__)


def load_image_from_url(img_url):
    response = requests.get(img_url)
    image_bytes = io.BytesIO(response.content)
    return image_bytes


@app.route("/predict", methods=['POST'])
def predict():
    input_json = request.get_json()
    if input_json is not None:
        print("Received following request: " + str(input_json))
        content_path = str(input_json['contentPath'])
        start_frame = int(input_json['startFrame'])
        end_frame = int(input_json['endFrame'])
        score = main_predictor.predict_score(content_path, start_frame, end_frame)
        print('predicted image score: ' + str(score))
        return jsonify({'aesthetic score': float(score)})
    return jsonify({'aesthetic score': -1})


app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
main_predictor = MainPredictor()

print("Aesthetic Predictor Initialized")

if __name__ == '__main__':
    try:
        print("Starting server...")
        app.run(debug=False, threaded=True, port=5001, host='0.0.0.0')
    except:
        print("Exception Occurred!")
