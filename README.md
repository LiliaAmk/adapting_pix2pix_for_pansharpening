# 🛰️ Adapting Pix2Pix for Pansharpening

This project explores the adaptation of the **Pix2Pix GAN** architecture for **pansharpening**, a task in remote sensing that fuses a high-resolution panchromatic image with a low-resolution multispectral image to produce a high-resolution multispectral output. The approach leverages **conditional GANs** to learn the mapping between input pairs and their sharpened output.

---

## 🎯 Objective

- Improve the spatial resolution of multispectral images using the structural detail from panchromatic bands.
- Train a deep learning model that learns this fusion in a supervised manner.

---

## 🧪 Workflow Summary

1. **Data Preparation**
   - Load paired datasets: PAN (grayscale, high-res) + MS (color, low-res)
   - Normalize and resize images to standard input shape

2. **Model Architecture**
   - Adapted **Pix2Pix** (U-Net generator + PatchGAN discriminator)
   - Inputs: Concatenated PAN and upsampled MS images
   - Output: Pansharpened MS image (same spatial resolution as PAN)

3. **Training**
   - Loss: Combination of adversarial loss + L1 loss for fidelity
   - Optimization with Adam
   - Model evaluation with visual outputs and quantitative metrics (e.g., PSNR)

4. **Results**
   - Visual comparisons between input, ground truth, and generated pansharpened outputs
   - Metrics like PSNR and SSIM for performance assessment

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, OpenCV
- Matplotlib, tqdm

---

## 📁 Files

- `adapting_pix2pix_for_pansharpening.ipynb`: Full pipeline from data loading to training and evaluation

---

## 🚀 How to Run

1. Install dependencies:
   pip install tensorflow opencv-python matplotlib numpy
