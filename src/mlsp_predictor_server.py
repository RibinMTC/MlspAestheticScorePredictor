from flask import Flask, jsonify, request

from werkzeug.middleware.proxy_fix import ProxyFix

from src.main_predictor import MainPredictor

model_initialized = False
app = Flask(__name__)


@app.route('/is_model_ready', methods=['GET'])
def is_model_ready():
    if model_initialized:
        return "Model initialized.", 200
    else:
        return "Model not initialized.", 400


@app.route("/predict", methods=['POST'])
def predict():
    if not model_initialized:
        return "Model not initialized.", 400
    input_json = request.get_json()
    if input_json is not None:
        print("Received following request: " + str(input_json))
        content_path = str(input_json['contentPath'])
        start_frame = int(input_json['startFrame'])
        end_frame = int(input_json['endFrame'])
        return main_predictor.predict(content_path, start_frame, end_frame)
    return "Error in prediction.", 400


app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
main_predictor = MainPredictor()

print("Aesthetic Predictor Initialized")
model_initialized = True

if __name__ == '__main__':
    try:
        print("Starting server...")
        app.run(debug=False, threaded=True, port=5001, host='0.0.0.0')
    except:
        print("Exception Occurred!")
