from PIL import Image
import math
import torch
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import os
from typing import List
#from groundingdino.util.inference import predict, load_image, load_model, Model
from groundingdino.util.inference import Model
#import supervision as sv
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

def enhance_class_name(class_names: List[str]) -> List[str]:
    print('enhance')
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def dino_detect(image_path):
	print('detect')
	# load image
	image = cv2.imread(image_path)

	# Define model
	config_path = r"groundingdino/config/GroundingDINO_SwinT_OGC.py"
	weights_path = r"groundingdino/weights/groundingdino_swint_ogc.pth"
	
	grounding_dino_model = Model(model_config_path=config_path, model_checkpoint_path=weights_path)
	
	# Define classes and thresholds
	CLASSES = ['amazon package']
	BOX_TRESHOLD = 0.35
	TEXT_TRESHOLD = 0.25

	# detect objects
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

def run_SAM(image_path,bbox):
	print('bbox',bbox.shape)
	SAM_ENCODER_VERSION = "vit_h"
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	sam_weights_path = r"segment-anything/weights/sam_vit_h_4b8939.pth"
	sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=sam_weights_path).to(device=DEVICE)
	sam_predictor = SamPredictor(sam)
	mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
    xyxy=bbox
	)
	print('shape: ',mask.shape)
	
	return mask


def overlay(old_path,new_path,mask,count):
	mask = np.squeeze(mask).astype(np.uint8)
	print(mask.shape)
	# visualize the predicted masks
	#plt.imshow(mask, cmap='gray')
	#plt.show()
	foreground = Image.open(old_path).convert('RGB')
	background = Image.open(new_path).convert('RGB')
	height, width = mask.shape
	foreground = foreground.resize((width, height))
	background = background.resize((width, height))
	mask = cv2.resize(mask, (width, height))

	# Convert the images to numpy arrays
	#mask = np.array(mask)
	foreground = np.array(foreground)
	background = np.array(background)
	# Create an alpha channel for the mask
	alpha = np.expand_dims(mask, axis=-1)

	# Create a new image by combining the foreground and background using the mask as the alpha channel
	new_image = np.where(alpha == 1, foreground, background)

	# Convert the new image to a PIL Image and display it
	new_image = Image.fromarray(new_image.astype(np.uint8))
	#new_image.show()
	output_path = r"../../Pictures/Synthetic_AmazonSBS/"
	new_image.save(output_path + "synthetic_blender" + str(count) + ".png")
	print('Saving image to',output_path+ "synthetic_blender" + str(count) + ".png")





if __name__ == '__main__':

	#

	# Define path to image with box
	image_path = r"/home/aaron/Pictures/amazonsbs_test"
	conveyor_image = r"/home/aaron/Pictures/Blender_Conveyors/shiny_roller5_withnoise2.png"

	# Create empty cartel json to store labels in
	cartel_json = {
    "categories": {
        "0": {}
    },
    "samples": {}
	}

	count = 0
	for image in os.listdir(image_path):
		#Detect with GroundingDINO
		bbox = dino_detect(image_path + "/" + image)
		# Store label in Cartel json
		cartel_json["samples"][str(count)] = {
			"bboxes": [
				{
					"angle": 0,
					"category_id": "0",
					"center_x": bbox.center_x,
					"center_y": bbox.center_y,
					"height": bbox.height,
					"width": bbox.width
				}
			],
			"image_id": image
		}
		
		# Pass detection to SAM
		mask = run_SAM(image_path + "/" + image,bbox)
		mask = mask.sum(axis=0)

		
		# Overlay the masked part of image onto conveyor
		overlay(image_path + "/" + image,conveyor_image,mask,count)
		count = count + 1
