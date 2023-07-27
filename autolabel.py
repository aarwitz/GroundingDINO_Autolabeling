"""
Input a folder of images and a natural language prompt 
Output a folder with rotated bounding boxes around prompted items and a json storing position and class of bbox

Uses an adapted cv2.minAreaRect() to create rotated bounding boxes of objects beyond the image's edge
    #           . P
    #          /|\
    #        /  | \
    # -----/----.--\--------------------
    # |  /      B   \                  |
    # |  \           \ Q               |
    # |   \         /                  |
    # |    \      /                    |
    # |     \   /                      |
    # |      \/                        |
    # |                                |
    # |                                |
    # ----------------------------------

"""

from pathlib import Path
import math
import torch
import cv2

import os
from typing import List
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import json
import shutil
from min_in_image_area_rect import min_in_image_area_rect
import supervision as sv

# Create empty cartel json to store labels in
cartel_json = {
"categories": {
    "0": {}
},
"samples": {}
}


def enhance_class_name(class_names: List[str]) -> List[str]:
    print('enhance')
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def dino_detect(image: np.ndarray) -> np.ndarray:
    print('detect')

    # Define model
    config_path = r"groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = r"groundingdino/weights/groundingdino_swint_ogc.pth"

    grounding_dino_model = Model(model_config_path=config_path, model_checkpoint_path=weights_path)

    # Define classes and thresholds

    CLASSES = ['white and black bags with white labels and black barcodes on their surface, all on a white, checkered bombay sorter.']
   # CLASSES = ['package . box . envelope . bag . mail . product . item . ' ]
   # CLASSES = ['package . brown box . parcel . orange envelope . brown envelope . shipments . bundels . cartons . box . bag . mail . product . item . toy . grocery .' ]
    # CLASSES = ['amazon package. parcel . box . mail . envelope . mail . product . tote . bag . grocery . bin .']
    # CLASSES = ['orange envelope package.']
    # CLASSES = ['package . box . ']
    # CLASSES = [' package . bag . envelope . parcel . box . mail . jiffy bag . poly bag . ']

    BOX_TRESHOLD = 0.32
    TEXT_TRESHOLD = 0.32
    # detect object
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    return detections.xyxy	

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def run_SAM(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
	SAM_ENCODER_VERSION = "vit_h"
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	sam_weights_path = r"segment-anything/weights/sam_vit_h_4b8939.pth"
	sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=sam_weights_path).to(device=DEVICE)
	sam_predictor = SamPredictor(sam)
	mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=bbox
	)
	return mask


def display_autolabel(new_image_path: str, labeled_image_path: str, bboxes: np.ndarray) -> None:
	"""
	Takes in image_path (image to be autolabeled) and bbox (generated autolabels from groundingdino)
	Writes a image with rectangle overlay to ./image_labeled
	"""
	# Get the generated image to be autlokabeled
	gen_image = cv2.imread(new_image_path)
	for bbox in bboxes:
		cv2.rectangle(gen_image, (bbox[0], bbox[1]), (bbox[2],bbox[3]), (0, 255, 0), 2)
	cv2.imwrite(labeled_image_path,gen_image)
	
def add_dino_autolabel_to_json(box_info: list, count: int, image_fname: str) -> None:
    bbox_list = [] # empty list for bboxes
    for bbox in box_info:
        center_x, center_y, angle, width, height = bbox
        bbox_data = {
            "angle": str(np.deg2rad(angle)),
            "category_id": "0",
            "center_x": str(center_x),
            "center_y": str(center_y),
            "height": str(height),
            "width": str(width)
        }
        bbox_list.append(bbox_data)
    cartel_json["samples"][str(count)] = {
        "bboxes": bbox_list,
        "image_id": str(count) + '.BMP'
    }
		

def rotate_bbox(image: np.ndarray, masks: np.ndarray, image_fname: str, output_path: Path, count: int) -> np.ndarray:
    #mport matplotlib.pyplot as plt
    box_info_4json = []
    cv2.imwrite(str(output_path / "images" / (str(count) + ".BMP")), image)
    for mask in masks:
        # get white pixels in mask
        coords = np.column_stack(np.where(mask.transpose() > 0))
        coords = coords.astype(np.int32)
        # get rotated rectangle that bounds the mask
        # rotrect = cv2.minAreaRect(coords)
        height, width, _ = image.shape
        rotrect = min_in_image_area_rect(coords, (width, height))
        # rotated rectangle box points
        box = np.int0(cv2.boxPoints(rotrect))
        # Draw the rotated rectangle on the original image
        cv2.drawContours(image, [box], 0, (0,0,255), 4)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box_info_4json += [get_rotated_bounding_box_info(box) ]
    #cv2.imwrite(str(output_path / "images_labeled" / (str(count) + ".BMP")), image[:, :, ::-1])
    cv2.imwrite(str(output_path / "images_labeled" / (str(count) + ".BMP")), image)
    return box_info_4json


def label_image_with_class(image: np.ndarray, detections, mask):
    CLASSES = ['package . box . mail . bag . product . item .' ]
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
          f"{CLASSES[class_id]} {confidence:0.2f}"
          for _, _, confidence, class_id, _
          in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    sv.plot_image(annotated_image, (16, 16))


def get_rotated_bounding_box_info(box):
    # Calculate the center point of the bounding box
    center_x = np.mean(box[:, 0])
    center_y = np.mean(box[:, 1])

    # Calculate the angle of rotation
    dx = box[1, 0] - box[0, 0]
    dy = box[1, 1] - box[0, 1]
    angle = np.arctan2(dy, dx) * (180 / math.pi)

    # Calculate the width and height of the bounding box
    width = np.linalg.norm(box[1] - box[0])
    height = np.linalg.norm(box[2] - box[1])

    return center_x, center_y, angle, width, height

def create_output_directory(output_path: Path) -> None:
    if os.path.exists(str(output_path)):
        shutil.rmtree(str(output_path))
    # Create the base directory 
    output_path.mkdir(parents=False, exist_ok=False)
    # Create labels folder
    (subdirectory1_path := output_path / "labels").mkdir(parents=True, exist_ok=False)
    # Create images folder
    (subdirectory2_path := output_path / "images").mkdir(parents=True, exist_ok=False)
    # Create labeled_images folder
    (subdirectory2_path := output_path / "images_labeled").mkdir(parents=True, exist_ok=False)

def main(image_path: Path, output_path: Path) -> None:
    # create output directory structure
    create_output_directory(output_path=output_path)
    count = 0
    for image_fname in os.listdir(image_path):
        # Load image
        image = cv2.imread(str(image_path / image_fname))
        # Detect with GroundingDINO
        bboxes = dino_detect(image)
        # Pass detection to SAM
        masks = run_SAM(image,bboxes)
        # Rotate bboxes using masks and write annotation to image and save
        box_info = rotate_bbox(image, masks, image_fname, output_path, count)
        # get the top right and bottom left coordinates
        label_image_with_class(image, bboxes, masks)
        # for box in boxes:
        #     boxes_labels = get_top_right_bottom_left
        # Store label in Cartel json
        add_dino_autolabel_to_json(box_info, count, image_fname)
        count+=1
    # Write json
    json_output_path = output_path / "labels" / "data.json"
    with open(output_path / "labels" / "data.json", 'w') as file:
        json.dump(cartel_json, file, indent=4)
    print(f"Data saved to {json_output_path} as JSON.")

if __name__ == '__main__':
    # Define path to a folder containing only images
    image_path = Path(r"/home/aaron/Pictures/NEW/FedexBombay")
    # Define path to output images, labeled images, and labels (data.json) to
    output_path = Path(r"/home/aaron/Pictures/NEW/FedexBombay_Labeled")
    # Call main function to autolabel images in image_path
    main(image_path, output_path)
    # Define path to output images, labeled images, and labels (data.json) to
    output_path = Path(r"/home/aaron/Pictures/NEW/FSRI_4_Labeled")
    # Call main function to autolabel images in image_path
    main(image_path, output_path)
