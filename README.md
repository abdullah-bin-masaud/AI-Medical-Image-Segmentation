# AI-Based Medical Image Segmentation System

A Deep Convolutional Neural Network (U-Net) pipeline for automated region-of-interest (ROI) segmentation of pathological lesions in axial brain MRI scans.

## Architecture
```
Contracting (Encoder)                     Expansive (Decoder)
[ 1x128x128 ]                             [ 1x128x128 Segmentation Mask ]
      |                                                ^
   Conv x2 ------------ (Skip Connection) ---------> Conv x2
      v                                                |
   MaxPool                                           Upsample
   Conv x2 ------------ (Skip Connection) ---------> Conv x2
      v                                                |
   MaxPool                                           Upsample
   Conv x2 ------------ (Skip Connection) ---------> Conv x2
      v                                                |
   MaxPool ------------------------------------------ Bottleneck
```

## Features
- **Encoder-Decoder Architecture:** Lightweight U-Net with contracting feature extraction and expansive skip-connected reconstruction.
- **Biomedical Loss & Metrics:** Formulated with Binary Cross Entropy (BCE) and evaluated using **Dice Similarity Coefficient** and **Intersection over Union (IoU)**.
- **Synthetic MRI Generator:** Procedurally generates realistic tissue parenchyma, ventricles, skull borders, and randomized pathological lesions with Rician noise.
- **Visual Diagnostics:** Automated side-by-side visualization of raw scans, anatomical ground truth, and thresholded network inferences.

## Quickstart
```bash
pip install -r requirements.txt
python train_evaluate.py
```
