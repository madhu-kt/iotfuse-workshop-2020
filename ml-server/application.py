# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import redis
import pickle
import uuid
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

ML_API_PREFIX = "ENTER ML SERVICE API PREFIX HERE"
REDIS_HOST = "ENTER REDIS HOST HERE"
REDIS_PASSWORD = "ENTER REDIS PASSWORD HERE"
redis_conn = redis.Redis(host = REDIS_HOST, port=6379, db=0, password=REDIS_PASSWORD)

@app.route('/', methods=['GET','POST'])
@cross_origin()
def about():
    return 'IoTFuse 2020 ML Deployment Service'

@app.route('/uploadCount/<app_id>', methods=['GET'])
@cross_origin()
def get_upload_count(app_id):
    upload_count = redis_conn.hget(app_id,"upload_count")
    if upload_count:
        upload_count = int(upload_count)
    else:
        upload_count = 0
    return jsonify({"count":upload_count})

@app.route('/deploy/<app_id>', methods=['POST'])
@cross_origin()
def deploy_model(app_id):
    f = request.files["model"]
    f_out_path = '/tmp/{app_id}.model'.format(app_id=app_id)
    model = f.save(f_out_path)
    return jsonify({"status":"Your model was successfully deployed!","endpoint":"{ml_api_prefix}/predict/{app_id}".format(ml_api_prefix=ML_API_PREFIX, app_id=app_id)})

# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

@app.route('/predict/<app_id>', methods=['POST'])
@cross_origin()
def predict(app_id):
    try:
        im_str = request.json
        im_str = im_str.split('data:image/jpeg;base64,')[1]
        
        print("[INFO] loading pre-trained network...")

        model_path = '/tmp/{app_id}.model'.format(app_id=app_id)
        
        model = load_model(model_path)

        image = stringToRGB(im_str)
        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image_input = np.array([image])
        pred = model.predict(image_input)
        predIdxs = np.argmax(pred, axis=1)
        predX = "AT RISK" if predIdxs[0] == 0 else "NORMAL"

        # increment redis counter
        redis_conn.hincrby(app_id,"upload_count",1)
        return jsonify({
            "result": {
                "prediction": predX,
                "probability": round(float(np.max(pred,axis=1)[0]),2)
            }
        })
    except Exception as e:
        return jsonify({
            "error": "Invalid API call.",
            "error_trace": str(e)
        })