from PIL import Image
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def remove_padding_from_images(input_folder, output_folder, padding=300):
    # Create the output folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(str(output_folder))
    os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if not f.endswith('.xml')]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        xml_file = os.path.join(input_folder, os.path.splitext(image_file)[0] + ".xml")
        
        # Open the image using PIL
        image = Image.open(image_path)

        # Get the original image dimensions
        original_width, original_height = image.size

        # Calculate the new dimensions without padding
        new_width = original_width - 2 * padding
        new_height = original_height - 2 * padding

        # Create a new image with the specified dimensions and fill it with a transparent background
        new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

        # Calculate the position to paste the central part of the original image onto the new image
        offset_x = padding
        offset_y = padding

        # Paste the central part of the original image onto the new image
        new_image.paste(image.crop((offset_x, offset_y, offset_x + new_width, offset_y + new_height)))

        # Save the new image without padding to the output folder
        output_image_path = os.path.join(output_folder, image_file)
        new_image.save(output_image_path)

        # Process the XML label
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall('object'):
            robndbox = obj.find('robndbox')
            cx = float(robndbox.find('cx').text) - padding
            cy = float(robndbox.find('cy').text) - padding
            robndbox.find('cx').text = str(cx)
            robndbox.find('cy').text = str(cy)

        # Save the modified XML label to the output folder
        output_xml_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".xml")
        tree.write(output_xml_path)

if __name__ == "__main__":
    # Set the input and output folders here
    input_folder = Path("/home/g5_team3/_aaron/Autolabeling/Autolabel_Resized_Test")         # Input folder with padded images and labels
    # output_folder is created by this script and new .xml's and images are inserted
    output_folder = Path("/home/g5_team3/Results/Cartel_Singulated v6")  # Output folder for images and labels without padding

    # Remove padding from the images and adjust corresponding XML labels
    remove_padding_from_images(input_folder, output_folder)