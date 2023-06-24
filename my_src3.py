from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os

model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
#IMAGE_PATH = "data/dog-3.jpeg"
#TEXT_PROMPT = "chair . person . dog ."
IMAGE_PATH = "SBS/"
TEXT_PROMPT = "box . bag . black conveyor belt ."
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25

for image_fname in os.listdir(IMAGE_PATH):
	if not (image_fname.endswith(".jpg")):
		print('skip')
		continue
	image_source, image = load_image(IMAGE_PATH+image_fname)
	boxes, logits, phrases = predict(
	model=model,
	image=image,
    	caption=TEXT_PROMPT,
    	box_threshold=BOX_TRESHOLD,
    	text_threshold=TEXT_TRESHOLD
	)

	annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
	cv2.imwrite(image_fname, annotated_frame)
