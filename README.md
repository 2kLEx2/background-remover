Background Remover
This Python script automatically removes the background from images using the RMBG-1.4 model. It continuously monitors a specified folder for new images, processes them, and saves the results with transparent backgrounds.

Features
Automatically detects and processes new images in the input folder.
Removes backgrounds using the RMBG-1.4 deep learning model.
Saves images with transparent backgrounds (.png format).
Supports GPU acceleration for faster processing.
Prerequisites
Python 3.8 or higher
NVIDIA GPU (optional, for acceleration with CUDA)
Installation
1. Clone the Repository
bash
Kopieren
Bearbeiten
git clone https://github.com/2kLEx2/background-remover.git
cd background-remover
2. Install Dependencies
bash
Kopieren
Bearbeiten
pip install -r requirements.txt
3. Set Up Configuration
Edit the config.json file to specify the input and output folders:

json
Kopieren
Bearbeiten
{
  "input_folder": "path/to/input/folder",
  "output_folder": "path/to/output/folder"
}
Replace "path/to/input/folder" and "path/to/output/folder" with the actual paths on your system.

Usage
Running the Script
bash
Kopieren
Bearbeiten
python image_processor.py
The script will monitor the input_folder for new images and process them automatically.

GPU Acceleration
To enable GPU support, ensure you have:

An NVIDIA GPU with CUDA support.

PyTorch with CUDA installed. If not installed, run:

bash
Kopieren
Bearbeiten
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
The script will automatically use the GPU if available; otherwise, it falls back to CPU.

Processing Images
Place images (.jpg, .png, .jpeg) in the input_folder.
The script detects new images and removes their backgrounds.
The processed images are saved in the output_folder with transparent backgrounds.
