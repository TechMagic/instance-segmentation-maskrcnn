import os
import cv2
import math
import flask
import logging
import numpy as np
import tensorflow as tf

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from flask import request, jsonify

batch_size = 1
app = flask.Flask(__name__)
model, graph, session = None, None, None

logging.basicConfig()
model_shape = (2048, 1280)


class InferenceConfig(Config):

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Give the configuration a recognizable name
    NAME = 'sketch'

    BACKBONE = 'resnet50'

    USE_MINI_MASK = True

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + measurement + parcel number

    # Use small images for faster training. Set the limits of the small side
    # the large side, which determines the image shape.
    IMAGE_MIN_DIM = 1280
    IMAGE_MAX_DIM = 2048

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 24, 32, 40)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 300


@app.route('/recognize', methods=['POST'])
def recognize():
    file_keys = list(request.files.keys())
    batch_count = math.ceil(len(file_keys) / batch_size)

    if batch_count:
        result = []

        for batch in range(batch_count):
            start = batch * batch_size
            stop = start + batch_size

            images, original_shapes = [], []
            for key in file_keys[start:stop]:
                image = cv2.imdecode(np.frombuffer(request.files[key].read(), np.uint8), cv2.IMREAD_COLOR)
                original_shapes.append(image.shape)

                resized = cv2.resize(image, (model_shape[1], model_shape[0]))
                images.append(resized)

            with graph.as_default():
                with session.as_default():
                    detections = model.detect(images, verbose=0)

            for i, (detection, original_shape) in enumerate(zip(detections, original_shapes)):
                d_masks = detection['masks']
                if d_masks.shape[0]:
                    masks = []
                    shape = (original_shape[1], original_shape[0])
                    for j in range(d_masks.shape[2]):
                        mask_nonzero = np.nonzero(cv2.resize(d_masks[:, :, j].astype(np.uint8), dsize=shape))
                        masks.append([mask_nonzero[0].tolist(), mask_nonzero[1].tolist()])

                    class_ids = detection['class_ids'].tolist()
                else:
                    masks, class_ids = [], []

                result.append({
                    'masks': masks,
                    'class_ids': class_ids
                })

        result = {
            'success': True,
            'result': result
        }
    else:
        result = {
            'success': False
        }

    return jsonify(result)


@app.before_first_request
def before_first_request():
    global model, graph, session

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        with session.as_default():
            model = MaskRCNN(mode='inference', config=InferenceConfig(), model_dir='')

            weight_location = os.environ.get('WEIGHTS', 'weights/model_weights.h5')
            model.load_weights(weight_location, by_name=True)


if __name__ == '__main__':
    port = os.environ.get('PORT', 5003)
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, use_reloader=False)
