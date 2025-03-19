import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from torchvision.transforms.functional import normalize

with open("config.json", "r") as config_file:
    config = json.load(config_file)

INPUT_FOLDER = config["input_folder"]
OUTPUT_FOLDER = config["output_folder"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    """Preprocess the image for the RMBG-1.4 model."""
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    """Postprocess the model's output to create a mask."""
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

def remove_background(image_path, output_path):
    """Remove the background from an image and save the result."""
    try:
        orig_image = Image.open(image_path).convert("RGB")
        orig_im = np.array(orig_image)
        orig_im_size = orig_im.shape[0:2]

        model_input_size = [1024, 1024]
        image = preprocess_image(orig_im, model_input_size).to(device)

        with torch.no_grad():
            result = model(image)

        result_image = postprocess_image(result[0][0], orig_im_size)

        pil_mask_im = Image.fromarray(result_image)

        no_bg_image = orig_image.copy()
        no_bg_image.putalpha(pil_mask_im)

        output_path = os.path.splitext(output_path)[0] + ".png"
        no_bg_image.save(output_path)
        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

class ImageHandler(FileSystemEventHandler):
    """Handler for new image files."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".jpg", ".png", ".jpeg")):
            print(f"New image detected: {event.src_path}")
            output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(event.src_path))
            remove_background(event.src_path, output_path)

def start_monitoring():
    """Start monitoring the input folder for new images."""
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=INPUT_FOLDER, recursive=False)
    observer.start()
    print(f"Monitoring folder: {INPUT_FOLDER}")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring()
