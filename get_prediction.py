import requests
import json

import numpy as np
from PIL import Image


# server URL
url = 'http://localhost:8501/v1/models/img_classifier:predict'
image_size = (32, 32)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# img_path = 'static/truck.png'


def make_prediction(image_path):

    """
    Load image and preproces it for network
    :param image_path: path to image given
    :return: class name of the prediction
    """

    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0) / 255.0
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    predictions = np.argmax(predictions)
    return class_names[predictions]


