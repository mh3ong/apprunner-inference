import argparse

from flask import Flask
import tensorflow as tf
import numpy as np
import shutil
import os
import time

from tensorflow.keras.applications import (
    mobilenet
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mobilenet,mobilenet_v2,inception_v3', type=str)
parser.add_argument('--hostname', default='0.0.0.0', type=str)
parser.add_argument('--port', default=5001, type=int)
args = parser.parse_args()
models_to_load = args.model.split(',')
hostname = args.hostname
port = args.port

models = {
    'mobilenet': mobilenet,
    'mobilenet_v2': mobilenet_v2,
    'inception_v3': inception_v3
}

models_detail = {
    'mobilenet': mobilenet.MobileNet(weights='imagenet'),
    'mobilenet_v2': mobilenet_v2.MobileNetV2(weights='imagenet'),
    'inception_v3': inception_v3.InceptionV3(weights='imagenet')
}



def mobilenet_load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[224, 224])


def image_to_array(image):
    return tf.keras.preprocessing.image.img_to_array(image, dtype=np.int32)


def image_preprocess(image_array, model_name):
    return models[model_name].preprocess_input(
        image_array[tf.newaxis, ...])


def save_model(model, saved_model_dir):
    model = models_detail[model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    model.save(saved_model_dir, include_optimizer=False, save_format='tf')

loaded_models = {}

for model_name in models_to_load:
  model_names = models_detail.keys()
  if model_name in model_names:
      model_path = f'{model_name}_saved_model'
      if os.path.isdir(model_path) == False:
          print('model save')
          save_model(model_name, model_path)
      loaded_models[model_name] = tf.keras.models.load_model(model_path)
  else:
      print(f'model names must be in {model_names}')
      exit(1)

mobilenetv1_image_path = './dataset/imagenet/imagenet_1000_raw/n02782093_1.JPEG'
mobilenetv1_test_image = mobilenet_load_image(mobilenetv1_image_path)
mobilenetv1_test_image_array = image_to_array(mobilenetv1_test_image)
mobilenetv1_test_image_preprocessed = image_preprocess(mobilenetv1_test_image_array, 'mobilenet')


print('saving and loading models completed!\n')

app = Flask(__name__)

@app.route('/mobilenet')
def mobilenetv1():
    inference_start_time = time.time()
    result = loaded_models['mobilenet'].predict(mobilenetv1_test_image_preprocessed)
    inference_end_time = time.time()

    inference_time = inference_end_time - inference_start_time
    
    return f'mobilenetv1 inference success\ninference time:{inference_time}\n'

@app.route('/healthcheck')
def healthcheck():
    return "healthcheck page"

app.run(host=hostname, port=port, threaded=False)
