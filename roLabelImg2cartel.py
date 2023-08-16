import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any



"""
Convert from Imagenet's bounding box labeling format to Cartel format
"""


def convert_xml_to_json(xml_path: Path) -> Dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = {
        "categories": {"0": {}},
        "samples": {}
    }

    image_id = root.find("filename").text.split(".")[0]+ '.BMP'
    sample = {"bboxes": [], "image_id": image_id}

    for obj in root.findall("object"):
        bbox = obj.find("robndbox")
        bbox_data = {
            "angle": float(bbox.find("angle").text),
            "category_id": "0",  # Assuming a fixed category ID
            "center_x": float(bbox.find("cx").text),
            "center_y": float(bbox.find("cy").text),
            "width": float(bbox.find("w").text),
            "height": float(bbox.find("h").text)
        }
        sample["bboxes"].append(bbox_data)

    annotations["samples"][image_id] = sample
    return annotations


def convert_folder_to_json(xml_folder: Path, output_path: Path) -> None:
    annotations = {"categories": {"0": {}}, "samples": {}}

    xml_files = xml_folder.glob("*.xml")
    for xml_file in xml_files:
        image_annotations = convert_xml_to_json(xml_file)
        annotations["samples"].update(image_annotations["samples"])

    with output_path.open("w") as json_file:
        json.dump(annotations, json_file, indent=2)


def copy_images(xml_folder: Path, output_folder: Path, image_extension: str) -> None:
    image_files = xml_folder.glob(f"*{image_extension}")
    for image_file in image_files:
        shutil.copy2(image_file, output_folder)


def create_cartel_folder(cartel_path: Path) -> None:
    images_folder = cartel_path / "images"
    labels_folder = cartel_path / "labels"

    # Delete the existing cartel folder and its contents
    shutil.rmtree(cartel_path, ignore_errors=True)

    # Create the cartel folder and its subdirectories
    cartel_path.mkdir()
    images_folder.mkdir()
    labels_folder.mkdir()

    return images_folder, labels_folder

if __name__ == "main":
    xml_folder = Path("/home/g5_team3/_aaron/Cartel_Singulated v6")
    cartel_output_path = Path("/home/g5_team3/_aaron/cartel2")  # This folder will be generated with images and labels subdirectories
    image_extension = ".BMP"
    cartel_folder, labels_folder = create_cartel_folder(cartel_output_path)
    copy_images(xml_folder, cartel_folder, image_extension)
    convert_folder_to_json(xml_folder, labels_folder / "output.json")