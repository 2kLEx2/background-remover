# Image Background Remover

This Python script automatically removes the background from images using the RMBG-1.4 model. It monitors a specified folder for new images and processes them, saving the results with transparent backgrounds.

## Prerequisites

- Python 3.8 or higher

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image-background-remover.git
   cd image-background-remover
Install the required dependencies:

bash
Copy
pip install -r requirements.txt
Ensure you have the config.json file configured with the correct input and output folders:

json
Copy
{
    "input_folder": "path/to/input/folder",
    "output_folder": "path/to/output/folder"
}
Usage
Place the images you want to process in the input folder specified in config.json.

Run the script:

bash
Copy
python image_processor.py
The script will monitor the input folder for new images. When a new image is detected, it will process the image, remove the background, and save the result in the output folder as a PNG file with transparency.

Configuration
input_folder: The folder where the script will look for new images.

output_folder: The folder where the processed images will be saved.

Notes
The script supports JPG, JPEG, and PNG formats.

The output images will always be saved as PNG to preserve transparency.

Troubleshooting
Ensure that the paths in config.json are correct and accessible.

License
This project is licensed under the MIT License. See the LICENSE file for details.
