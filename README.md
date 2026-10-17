# Fingerprint Quality Analysis

I built this project to try out a straightforward image-processing workflow for fingerprint images. It uses Python and OpenCV to improve local contrast, reduce some image noise, estimate the fingerprint area, and calculate a few simple image measurements.

This is an educational project. It does not identify people, perform biometric matching, or calculate an official NFIQ or NFIQ 2 score. Its measurements are rough indicators, not a validated quality rating.

## What it does

You can run the pipeline on one image or on a folder of images. It accepts PNG, JPG, JPEG, BMP, TIF, and TIFF files. For each image, it:

- applies CLAHE to improve local contrast, then uses light denoising;
- estimates the foreground with Otsu thresholding and the largest connected component;
- calculates image dimensions, foreground ratio, Laplacian-variance sharpness, intensity-contrast, and foreground-gradient indicators;
- saves the enhanced image, estimated mask, a comparison figure, and per-image JSON metrics.

When processing a folder, it also writes a combined `quality_metrics.csv` file.

## Getting started

Python 3.10 or newer is recommended. From the project folder, create and activate a virtual environment, then install the dependencies.

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run it

Run the example using the bundled synthetic image:

```bash
python main.py
```

To use a single image or a directory of images, pass the input path. The output folder is created if needed:

```bash
python main.py --input path/to/fingerprint.png --output outputs
python main.py --input path/to/images --output outputs
```

The default input is `data/sample_images`, and the default output is `outputs`.

## Understanding the measurements

The measurements are easiest to use when comparing images captured under similar conditions. They can be affected by resolution, noise, lighting, cropping, and the foreground estimate, so a higher value does not automatically mean a better fingerprint image.

- **Laplacian variance** is a rough sharpness or focus indicator. Noise and fine texture can raise it too.
- **Intensity standard deviation** is a basic contrast indicator; it does not directly measure ridge clarity.
- **Foreground ratio** is the share of pixels included in the estimated mask. The simple segmentation can struggle with uneven lighting or tight crops.
- **Foreground gradient magnitude** measures local intensity changes in the estimated foreground. Both ridge edges and noise contribute to it.

## Images and privacy

The example image is synthetic. If you try the pipeline with other images, use only images you have permission to process. Fingerprints are sensitive biometric data, so think carefully before sharing them.

## Possible next steps

I’d like to compare these indicators with expert labels from a properly licensed dataset, test enhancement alongside a separately validated matcher, and explore better foreground segmentation and ridge-orientation visualisation.

## Built with

Python, OpenCV, NumPy, and Matplotlib.
