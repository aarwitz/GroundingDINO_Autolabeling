
from typing import Union, Callable, Optional, Any, Dict, List, Tuple


import json
import cv2
import numpy as np
from pathlib import Path

from tqdm import tqdm
import os
import shutil


def draw_bbox(np_image: np.ndarray, bbox, color) -> np.ndarray:
    """
    Return the original image with a bounding box draw on it
    """
    assert bbox["angle"] == 0, f"Please use draw_rotated_bbox for rotated boxes"
    x = bbox["center_x"] - bbox["width"] / 2
    y = bbox["center_y"] - bbox["height"] / 2
    start = (int(x), int(y))
    end = (int(x + bbox["width"]), int(y + bbox["height"]))
    np_image = cv2.rectangle(np_image, start, end, color, 2)
    return np_image


def draw_rotated_boxes(
    image: np.ndarray,
    box: List,
    copy: Optional[bool] = True,
    box_thickness: Optional[int] = 1,
    color: Optional[Tuple[int, int, int]] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw a set of rotated bounding boxes on an image

    image (ndarray): image to draw boxes on
    boxes (List[RawRotatedBoundingBox]): List of bounding boxes to draw
    copy (bool): If True copy the image before drawing to avoid modifying image in place
    box_thickness (int): thickness of drawn bounding box
    color (Tuple[int, ...]): Color to draw bounding boxes in
    text (List[str]): text to write for each bounding box.
        Should be same length as boxes and elementals convertible to string.
    """
    canvas = image.copy() if copy else image

    rotated_rects = []
    rotated_rect = (
        (float(box["center_x"]), float(box["center_y"])),
        (float(box["width"]), float(box["height"])),
        np.rad2deg(float(box["angle"])),
    )
    box_points = cv2.boxPoints(rotated_rect).astype(np.int64)
    rotated_rects.append(box_points)

    cv2.drawContours(canvas, rotated_rects, -1, color, thickness=box_thickness)
    return canvas



def visualize_annotations(image_directory: Path, labels_directory: Path, output_path: Path) -> None:
    # for each image, for each bbox, draw_bbox,writeimage
    if os.path.exists(str(output_path)):
        shutil.rmtree(str(output_path))
    output_path.mkdir()
    with open(str(labels_directory),'r') as labels:
        data = json.load(labels)
    for sample in tqdm(data["samples"]):
        np_img = cv2.imread(str(image_directory / data["samples"][sample]["image_id"]))
        for bbox in data["samples"][sample]["bboxes"]:
            if bbox["angle"] == 0:
                np_img = draw_bbox(np_img,bbox,(0,255,0))
            else:
                np_img = draw_rotated_boxes(image=np_img, box=bbox, box_thickness=2)
        cv2.imwrite(str(output_path / data["samples"][sample]["image_id"]), np_img)



def main(image_directory: Path,labels_directory: Path, output_path: Path):
    """
    View labels in labels_directory overlayed on image_directory
    """
    visualize_annotations(image_directory,labels_directory, output_path)




if __name__ == '__main__':
    # Directories for images and labels
    dataset_root = Path(r"/home/g5_team3/_aaron/cartel2")
    image_directory = dataset_root / "images"
    labels_directory = dataset_root / "labels" /"output.json"

    # Directory which will be created and images with overlayed labels will be saved to
    output_path = Path(r"/home/g5_team3/_aaron/visualize")

    main(image_directory, labels_directory, output_path)