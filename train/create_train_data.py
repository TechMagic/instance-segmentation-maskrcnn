import os
import cv2
import json
import shutil
import random
import zipfile
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

TARGET_SHAPE = (2048, 1280)
IN_DIR, OUT_DIR = 'C:/Users/FlorijnWim/Downloads/verwerkt', '../train_data'


def generate_box_mask(box, mask_shape):
    rct = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), box[2])
    corner_points = cv2.boxPoints(rct).astype(np.int32)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, corner_points, 1)

    mask = cv2.resize(mask, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)
    mask = np.nonzero(mask)
    assert len(mask[0])

    return mask


def generate_measurement_masks_from_json(obs, image_shape):
    dct = obs['text']

    measurement_dict = {}
    for key in dct:
        item = dct[key]
        if item['type'] == 'measurement':
            measurement_dict[key] = item

    masks = []
    for txt in measurement_dict:
        box = measurement_dict[txt]['box']
        masks.append(generate_box_mask(box, image_shape))

    return masks


def generate_parcel_number_masks_from_json(obs, image_shape):
    dct = obs['text']

    parcel_dict = {}
    for key in dct:
        item = dct[key]
        if item['type'] == 'parcel' and 'value' in item.keys():
            parcel_dict[key] = item

    masks = []
    for txt in parcel_dict:
        box = parcel_dict[txt]['box']
        masks.append(generate_box_mask(box, image_shape))

    return masks


def get_masked(img, measurement_masks, parcel_number_masks, visualize=False):
    alpha = 0.4
    masked = np.zeros(img.shape)
    mask_img = img.copy() if visualize else None

    if measurement_masks is not None:
        for mask in measurement_masks:
            masked[mask] = 1

            if visualize:
                mask_img[mask] = mask_img[mask] * [1., 2., 1.] * alpha + (1 - alpha)

    non_overlapping_parcel_number_masks = []
    if parcel_number_masks is not None:
        for mask in parcel_number_masks:
            non_overlapping_idx = np.where(masked[mask] == 0)[0]
            if len(non_overlapping_idx):
                mask = (mask[0][non_overlapping_idx], mask[1][non_overlapping_idx])
            masked[mask] = 1
            non_overlapping_parcel_number_masks.append(mask)

            if visualize:
                mask_img[mask] = mask_img[mask] * [2., 1., 1.] * alpha + (1 - alpha)

    if visualize:
        plt.figure(figsize=(4, 4))
        plt.imshow(mask_img)
        plt.show()

    return measurement_masks, non_overlapping_parcel_number_masks


def get_parcel_numbers(json_data):
    parcel_numbers = set()
    for text_id, data in json_data['text'].items():
        if data['type'] == 'parcel' and 'value' in data.keys():
            parcel_numbers.add(data['value'])

    return parcel_numbers


def read_zip(zip_name, measurement_masks=True, parcel_number_masks=True):
    with zipfile.ZipFile(zip_name, 'r') as archive:
        prefix, postfix = 'observations/snapshots/latest/', '.latest.json'
        sketch_files = list(filter(lambda x: x.startswith(prefix) and x.endswith(postfix), archive.namelist()))
        for i, sketch_file in enumerate(sketch_files):
            sketch_name = sketch_file[len(prefix):-len(postfix)]
            logging.info('Processing sketch: {index}: {name}.'.format(index=i, name=sketch_name))

            img_pf, img_extension = 'observations/attachments/front/' + sketch_name, '.JPG'
            img_files = list(filter(lambda x: x.startswith(img_pf) and x.endswith(img_extension), archive.namelist()))
            if len(img_files):
                file = archive.read(img_files[0])
                image = cv2.imdecode(np.frombuffer(file, np.uint8), 1)
                image_shape = image.shape[:2]

                image = cv2.resize(image, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)

                with archive.open(sketch_file, 'r') as fh:
                    json_data = json.loads(fh.read())

                if parcel_number_masks:
                    with archive.open('sketches/' + sketch_name + '/sketch.json', 'r') as fh:
                        meta_data = json.loads(fh.read())

                    parcel_numbers_vectorized = get_parcel_numbers(json_data)
                    parcel_numbers = set(map(lambda x: str(x['parcel']), meta_data['meta']['parcels']))
                    if not parcel_numbers == parcel_numbers_vectorized:
                        logging.info('Skipping {name}. Not all parcel numbers are present.'.format(name=sketch_name))
                        continue

                    p_masks = generate_parcel_number_masks_from_json(json_data, image_shape)
                else:
                    p_masks = None

                if measurement_masks:
                    m_masks = generate_measurement_masks_from_json(json_data, image_shape)

                    if not len(m_masks):
                        logging.info('Skipping {name}. No measurements are present.'.format(name=sketch_name))
                        continue
                else:
                    m_masks = None

                m_masks, p_masks = get_masked(image, m_masks, p_masks, visualize=False)

                base_dir = os.path.join(OUT_DIR, sketch_name)
                if os.path.exists(base_dir):
                    shutil.rmtree(base_dir)
                Path(base_dir).mkdir(parents=True)

                if measurement_masks:
                    measurement_output_path = os.path.join(base_dir, 'measurement_masks')
                    Path(measurement_output_path).mkdir(parents=True, exist_ok=True)

                    for index, mask in enumerate(m_masks):
                        np.save(os.path.join(measurement_output_path, '{i}.npy'.format(i=index)), mask)

                if parcel_number_masks:
                    parcel_output_path = os.path.join(base_dir, 'parcel_number_masks')
                    Path(parcel_output_path).mkdir(parents=True, exist_ok=True)

                    for index, mask in enumerate(p_masks):
                        np.save(os.path.join(parcel_output_path, '{i}.npy'.format(i=index)), mask)

                cv2.imwrite(os.path.join(base_dir, 'image.tif'), image)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    zip_file_names = os.listdir(IN_DIR)
    random.shuffle(zip_file_names)
    for j, name in enumerate(zip_file_names):
        logging.info('Processing project {index}: {name}.'.format(index=j, name=name))
        read_zip(os.path.join(IN_DIR, name))
