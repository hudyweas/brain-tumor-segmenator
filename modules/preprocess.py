import os
import numpy as np
import cv2
import json

def create_mask(img_data, segmentation, mask_output_folder):
    mask_np = np.zeros((img_data['height'], img_data['width']), dtype=np.uint8)

    for idx, seg in enumerate(segmentation):
        polygon = [(int(seg[i]), int(seg[i+1])) for i in range(0, len(seg), 2)]
        img = cv2.rectangle(mask_np, polygon[0], polygon[2], 255, thickness=-1)

        mask_output = os.path.join(mask_output_folder, img_data['file_name'])

        cv2.imwrite(mask_output, img)
    pass

def preprocess_data(input_path, output_path):
    images_output_folder = os.path.join(output_path + "_prep/", 'images')
    masks_output_folder = os.path.join(output_path + "_prep/", 'masks')

    #making sure that folders exists
    if not os.path.exists(masks_output_folder):
        os.makedirs(masks_output_folder)
    if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

    #getting annotations from the json file and extracting the images data
    annotations = json.load(open(os.path.join(input_path, '_annotations.coco.json')))
    annotations['annotations']

    segmentation_map = {}
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        segmentation_map[img_id] = segmentation_map[img_id].append(ann['segmentation']) if img_id in segmentation_map else ann['segmentation']

    for img_data in annotations['images']:
        #making sure that the image has annotations
        if img_data['id'] not in segmentation_map:
            continue
        if segmentation_map[img_data['id']] is None:
            continue

        #creating masks
        create_mask(img_data, segmentation_map[img_data['id']], masks_output_folder)

        #preprocessing images
        image = cv2.imread(os.path.join(input_path, img_data['file_name']))
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(images_output_folder, img_data['file_name']), image)
