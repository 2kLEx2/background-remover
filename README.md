# Image Background Remover

This Python script automatically removes the background from images using the RMBG-1.4 model. It monitors a specified folder for new images and processes them, saving the results with transparent backgrounds.

## Prerequisites

- Python 3.8 or higher

## Installation

1. Clone this repository:
   
   ```bash
   git clone https://github.com/2kLEx2/background-remover.git
   cd background-remover
   ```

2. Install the required dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Edit the `config.json` file to set the correct paths for your screenshot folder and output folder.

2. Run the script:
   
   ```bash
   python image_processor.py
   ```

The script will monitor the input folder for new images. When a new image is detected, it will process the image, remove the background, and save the result in the output folder as a PNG file with transparency.
