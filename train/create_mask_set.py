import os
import cv2
import json
import zipfile
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import transform

convex_hull_th = 200
target_shape = (1024, 680)


def generate_box_mask(box, mask_shape, image_gs):
    rct = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), box[2])
    corner_points = cv2.boxPoints(rct).astype(np.int32)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, corner_points, 1)
    mask = transform.resize(mask, output_shape=target_shape, mode='edge', order=0, preserve_range=True).astype(np.uint8)
    mask = np.nonzero(mask)

    zeros = np.zeros(image_gs.shape)
    zeros[mask] = image_gs[mask] < convex_hull_th
    mask = np.nonzero(zeros)

    image = np.zeros(image_gs.shape)
    if len(mask[0]):
        mask_points = np.hstack((mask[1][:, None], mask[0][:, None]))
        hull = cv2.convexHull(mask_points)
        cv2.drawContours(image, [hull], -1, 1, thickness=-1)

    mask = np.nonzero(image)

    return mask


def generate_line_mask(line, mask_shape, image_gs):
    mask = np.zeros(mask_shape, dtype='uint8')
    cv2.line(mask, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), 255, 3)
    mask = transform.resize(mask, output_shape=target_shape, mode='edge', order=0, preserve_range=True).astype(np.uint8)
    mask = np.nonzero(mask)

    zeros = np.zeros(image_gs.shape)
    zeros[mask] = image_gs[mask] < convex_hull_th
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


def generate_lines_from_json(obs, image_shape, image_gs):
    lines = obs['lines']
    points_dict = obs['points']

    lines_list = []
    for line, points in lines.items():
        for i, end in enumerate(points['points'][1:]):
            start = points['points'][i]
            lines_list.append([points_dict[start]['position'], points_dict[end]['position']])

    masks = list(map(lambda x: generate_line_mask(x, image_shape, image_gs), lines_list))

    return masks


def generate_building_from_json(obs, image_shape, image_gs):
    points_dict = obs['points']
    buildings = obs['buildings']

    buildings_list = []
    for building, points in buildings.items():
        buildings_list.append(list(map(lambda x: list(map(int, points_dict[x]['position'])), points['points'])))

    masks = list(map(lambda x: generate_building_mask(x, image_shape, image_gs), buildings_list))

    return masks


def get_masked(img, measurement_masks, parcel_number_masks, line_masks, building_masks, visualize=False):
    alpha = 0.4
    mask_img = img.copy()

    for mask in measurement_masks:
        mask_img[mask] = mask_img[mask] * [1., 2., 1.] * alpha + (1 - alpha)
    for mask in parcel_number_masks:
        mask_img[mask] = mask_img[mask] * [1., 2., 1.] * alpha + (1 - alpha)
    for mask in line_masks:
        mask_img[mask] = mask_img[mask] * alpha + (1 - alpha)
    for mask in building_masks:
        mask_img[mask] = mask_img[mask] * [2., 1., 1.] * alpha + (1 - alpha)

    if visualize:
        plt.figure(figsize=(4, 4))
        plt.imshow(mask_img)
        plt.show()

    return mask_img


def read_zip(zip_name):
    output_dir = '../traindata_maskrcnn'
    with zipfile.ZipFile(zip_name, 'r') as archive:
        prefix, postfix = 'observations/snapshots/latest/', '.latest.json'
        sketch_files = list(filter(lambda x: x.startswith(prefix) and x.endswith(postfix), archive.namelist()))
        for i, sketch_file in enumerate(sketch_files):
            sketch_name = sketch_file[len(prefix):-len(postfix)]

            logging.info('Processing: {index}: {name}.'.format(index=i, name=sketch_name))

            attachment_prefix, img_extension = 'observations/attachments/front_jpeg_removed/' + sketch_name, '.png'
            image_files = list(filter(lambda x: x.startswith(attachment_prefix) and x.endswith(img_extension),
                                      archive.namelist()))
            if len(image_files):
                image_file = image_files[0]
                file = archive.read(image_file)
                image = cv2.imdecode(np.frombuffer(file, np.uint8), 1)
                image_shape = image.shape[:2]

                image = transform \
                    .resize(image, output_shape=target_shape, mode='edge', order=3, preserve_range=True) \
                    .astype(np.uint8)

                base_dir = os.path.join(output_dir, sketch_name)
                Path(base_dir).mkdir(parents=True, exist_ok=True)

                fh = archive.open(sketch_file, 'r')
                json_data = json.loads(fh.read())

                image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                measurement_masks = generate_measurement_masks_from_json(json_data, image_shape, image_gs)
                parcel_number_masks = generate_parcel_number_masks_from_json(json_data, image_shape, image_gs)
                line_masks = generate_lines_from_json(json_data, image_shape, image_gs)
                building_masks = generate_building_from_json(json_data, image_shape, image_gs)

                cv2.imwrite(os.path.join(base_dir, 'image.png'), image)

                output_path = os.path.join(base_dir, 'measurement_masks')
                Path(output_path).mkdir(parents=True, exist_ok=True)
                for index, mask in enumerate(measurement_masks):
                    np.save(os.path.join(output_path, '{}.npy'.format(index)), mask)

                output_path = os.path.join(base_dir, 'parcel_number_masks')
                Path(output_path).mkdir(parents=True, exist_ok=True)
                for index, mask in enumerate(parcel_number_masks):
                    np.save(os.path.join(output_path, '{}.npy'.format(index)), mask)

                output_path = os.path.join(base_dir, 'line_masks')
                Path(output_path).mkdir(parents=True, exist_ok=True)
                for index, mask in enumerate(line_masks):
                    np.save(os.path.join(output_path, '{}.npy'.format(index)), mask)

                output_path = os.path.join(base_dir, 'building_masks')
                Path(output_path).mkdir(parents=True, exist_ok=True)
                for index, mask in enumerate(building_masks):
                    np.save(os.path.join(output_path, '{}.npy'.format(index)), mask)

                masked_img = get_masked(image, measurement_masks, parcel_number_masks, line_masks, building_masks)
                cv2.imwrite(os.path.join(base_dir, 'masked.png'), masked_img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    read_zip('DSMSOFT_001_VKB01-5e4e18f0756b351c979eb31f.zip')
