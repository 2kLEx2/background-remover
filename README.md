# Background Remover

This Python script automatically removes the background from images using the **RMBG-1.4** model. It continuously monitors a specified folder for new images, processes them, and saves the results with transparent backgrounds.

## Features

- Automatically detects and processes new images in the input folder.
- Removes backgrounds using the **RMBG-1.4** deep learning model.
- Saves images with transparent backgrounds (`.png` format).
- Supports **GPU acceleration** for faster processing.

## Prerequisites

- **Python 3.8 or higher**
- **NVIDIA GPU** (optional, for acceleration with CUDA)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/2kLEx2/background-remover.git
cd background-remover
