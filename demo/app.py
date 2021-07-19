import os
from flask import Flask, render_template, request, send_from_directory
from keras_preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'

global model

def load__model():
    """Load model once at running time for all the predictions"""
    print('[INFO] : Model loading ................')
    # model = tf.keras.models.load_model(MODEL_FOLDER + '/catsVSdogs.h5')
    global model
    model = load_model(MODEL_FOLDER + '/leaf2.hdf5')
    global graph
    graph = tf.compat.v1.get_default_graph()
    print('[INFO] : Model loaded')

def padding_images(img):

    desired_size = 256
    im = cv2.imread(img)
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


def predict(fullpath):

    data = image.load_img(fullpath, target_size=(256, 256, 3))
    # (150,150,3) ==> (1,150,150,3)
    data = np.expand_dims(data, axis=0)
    # Scaling
    # data = data.astype('float') / 255

    # Prediction

    with graph.as_default():
        global model
        model = load_model(MODEL_FOLDER + '/leaf2.hdf5')
        result = model.predict(data)

    return result


# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)
        img = Image.open(fullname)
        if img.size != (256, 256) :
            OUTPUT_PATH = './static/uploads'

            files = []
            # r=root, d=directories, f = files
            files.append(os.path.join(fullname))

            for i in tqdm(range(0, len(files))):
                img = padding_images(files[i])
                plt.imsave(os.path.join(OUTPUT_PATH, "new.jpg"), img)


            fullname = OUTPUT_PATH + "/" + "new.jpg"

        result = predict(fullname)

        pred_prob = result.item()

        if pred_prob > .5:
            label = 'lá xoài bệnh'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'lá xoài khỏe'
            accuracy = round((1 - pred_prob) * 100, 2)

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    load__model()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
