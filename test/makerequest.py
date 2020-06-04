import cv2
import json
import logging
import requests
import numpy as np


def test_recognize():
    url = 'http://127.0.0.1:5003/recognize'

    img = cv2.imread('../train_data/image.tif')
    _, img_encoded = cv2.imencode('.png', img)

    files = {
        'img.png': ('img.png', img_encoded.tostring(), 'image/png')
    }

    response = requests.post(url, files=files)
    result = json.loads(response.text)
    print(result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_recognize()
