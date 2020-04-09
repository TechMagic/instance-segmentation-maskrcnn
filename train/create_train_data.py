import os
import cv2
import json
import shutil
import zipfile
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import transform

HULL_TH = 200
TARGET_SHAPE = (1280, 768)

IN_DIR, OUT_DIR = 'C:/Users/FlorijnWim/Downloads/verwerkt', '../train_data'


def generate_box_mask(box, mask_shape, image_gs):
    rct = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), box[2])
    corner_points = cv2.boxPoints(rct).astype(np.int32)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, corner_points, 1)
    mask = transform \
        .resize(mask, output_shape=TARGET_SHAPE, mode='edge', order=0, preserve_range=True) \
        .astype(np.uint8)
    mask = np.nonzero(mask)

    zeros = np.zeros(image_gs.shape)
    zeros[mask] = image_gs[mask] < HULL_TH
    mask = np.nonzero(zeros)

    image = np.zeros(image_gs.shape)
    if len(mask[0]):
        mask_points = np.hstack((mask[1][:, None], mask[0][:, None]))
        hull = cv2.convexHull(mask_points)
        cv2.drawContours(image, [hull], -1, 1, thickness=-1)

    mask = np.nonzero(image)

    return mask


def generate_building_mask(building, mask_shape, image_gs):
    mask = np.zeros(mask_shape)
    if len(building):
        building = np.array([np.asarray(building)])
        cv2.fillPoly(mask, [building], 1)

    mask = transform.resize(mask, output_shape=image_gs.shape, mode='edge', order=0, preserve_range=True) \
        .astype(np.uint8)

    mask = np.nonzero(mask)

    return mask


def generate_measurement_masks_from_json(obs, image_shape, image_gs):
    dct = obs['text']

    measurement_dict = {}
    for key in dct:
        item = dct[key]
        if item['type'] == 'measurement':
            measurement_dict[key] = item

    masks = []
    for txt in measurement_dict:
        box = measurement_dict[txt]['box']
        masks.append(generate_box_mask(box, image_shape, image_gs))

    return masks


def generate_parcel_number_masks_from_json(obs, image_shape, image_gs):
    dct = obs['text']

    parcel_dict = {}
    for key in dct:
        item = dct[key]
        if item['type'] == 'parcel':
            parcel_dict[key] = item

    masks = []
    for txt in parcel_dict:
        box = parcel_dict[txt]['box']
        masks.append(generate_box_mask(box, image_shape, image_gs))

    return masks


def generate_building_from_json(obs, image_shape, image_gs):
    points_dict, buildings = obs['points'], obs['buildings']

    buildings_list = []
    for building, points in buildings.items():
        buildings_list.append(list(map(lambda x: list(map(int, points_dict[x]['position'])), points['points'])))

    masks = list(map(lambda x: generate_building_mask(x, image_shape, image_gs), buildings_list))

    return masks


def get_masked(img, measurement_masks, parcel_number_masks, building_masks, visualize=False):
    alpha = 0.4
    mask_img = img.copy()

    if measurement_masks is not None:
        for mask in measurement_masks:
            mask_img[mask] = mask_img[mask] * [1., 2., 1.] * alpha + (1 - alpha)

    if parcel_number_masks is not None:
        for mask in parcel_number_masks:
            mask_img[mask] = mask_img[mask] * [2., 1., 1.] * alpha + (1 - alpha)

    if building_masks is not None:
        for mask in building_masks:
            mask_img[mask] = mask_img[mask] * [1., 1., 2.] * alpha + (1 - alpha)

    if visualize:
        plt.figure(figsize=(4, 4))
        plt.imshow(mask_img)
        plt.show()

    return mask_img


def read_zip(zip_name, measurement_masks=True, parcel_number_masks=True, building_masks=True):
    with zipfile.ZipFile(zip_name, 'r') as archive:
        prefix, postfix = 'observations/snapshots/latest/', '.latest.json'
        sketch_files = list(filter(lambda x: x.startswith(prefix) and x.endswith(postfix), archive.namelist()))
        for i, sketch_file in enumerate(sketch_files):
            sketch_name = sketch_file[len(prefix):-len(postfix)]

            logging.info('Processing sketch: {index}: {name}.'.format(index=i, name=sketch_name))

            attachment_prefix, img_extension = 'observations/attachments/front_jpeg_removed/' + sketch_name, '.png'
            image_files = list(filter(lambda x: x.startswith(attachment_prefix) and x.endswith(img_extension),
                                      archive.namelist()))
            if len(image_files):
                file = archive.read(image_files[0])
                image = cv2.imdecode(np.frombuffer(file, np.uint8), 1)
                image_shape = image.shape[:2]

                image = transform \
                    .resize(image, output_shape=TARGET_SHAPE, mode='edge', order=3, preserve_range=True) \
                    .astype(np.uint8)

                base_dir = os.path.join(OUT_DIR, sketch_name)
                if os.path.exists(base_dir):
                    shutil.rmtree(base_dir)
                Path(base_dir).mkdir(parents=True)

                with archive.open(sketch_file, 'r') as fh:
                    json_data = json.loads(fh.read())
                image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if measurement_masks:
                    measurement_output_path = os.path.join(base_dir, 'measurement_masks')
                    Path(measurement_output_path).mkdir(parents=True, exist_ok=True)
                    m_masks = generate_measurement_masks_from_json(json_data, image_shape, image_gs)
                else:
                    m_masks = None

                if parcel_number_masks:
                    parcel_output_path = os.path.join(base_dir, 'parcel_number_masks')
                    Path(parcel_output_path).mkdir(parents=True, exist_ok=True)
                    p_masks = generate_parcel_number_masks_from_json(json_data, image_shape, image_gs)
                else:
                    p_masks = None

                if building_masks:
                    building_output_path = os.path.join(base_dir, 'building_masks')
                    Path(building_output_path).mkdir(parents=True, exist_ok=True)
                    b_masks = generate_building_from_json(json_data, image_shape, image_gs)
                else:
                    b_masks = None

                masked_img = get_masked(image, m_masks, p_masks, b_masks, visualize=False)

                if measurement_masks:
                    for index, mask in enumerate(m_masks):
                        np.save(os.path.join(measurement_output_path, '{i}.npy'.format(i=index)), mask)

                if parcel_number_masks:
                    for index, mask in enumerate(p_masks):
                        np.save(os.path.join(parcel_output_path, '{i}.npy'.format(i=index)), mask)

                if building_masks:
                    for index, mask in enumerate(b_masks):
                        np.save(os.path.join(building_output_path, '{i}.npy'.format(i=index)), mask)

                cv2.imwrite(os.path.join(base_dir, 'image.png'), image)
                cv2.imwrite(os.path.join(base_dir, 'masked.png'), masked_img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    for j, name in enumerate(os.listdir(IN_DIR)):
        logging.info('Processing project: {index}: {name}.'.format(index=j, name=name))
        read_zip(os.path.join(IN_DIR, name))
