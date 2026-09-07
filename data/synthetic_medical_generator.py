"""
Generates synthetic medical MRI slices with simulated anatomical structures and lesion masks.
"""
import numpy as np
import cv2
from pathlib import Path

DATA_DIR = Path(__file__).parent


def generate_synthetic_mri(num_samples: int = 40, img_size: int = 128):
    """Creates synthetic axial brain MRI slices with ground-truth lesion segmentations."""
    images = []
    masks = []

    for i in range(num_samples):
        # 1. Background skull/brain ellipse
        img = np.zeros((img_size, img_size), dtype=np.float32)
        mask = np.zeros((img_size, img_size), dtype=np.float32)

        # Outer skull
        cv2.ellipse(img, (img_size//2, img_size//2), (img_size//2 - 10, img_size//2 - 20), 0, 0, 360, 0.4, -1)
        # Brain parenchyma tissue
        cv2.ellipse(img, (img_size//2, img_size//2), (img_size//2 - 18, img_size//2 - 28), 0, 0, 360, 0.65, -1)
        # Ventricles (dark cerebrospinal fluid)
        cv2.ellipse(img, (img_size//2 - 12, img_size//2), (8, 18), 10, 0, 360, 0.2, -1)
        cv2.ellipse(img, (img_size//2 + 12, img_size//2), (8, 18), -10, 0, 360, 0.2, -1)

        # 2. Add randomized pathological lesion / tumor
        if np.random.rand() > 0.15:  # 85% have lesions
            lx = np.random.randint(img_size//2 - 25, img_size//2 + 25)
            ly = np.random.randint(img_size//2 - 25, img_size//2 + 25)
            lr = np.random.randint(6, 16)
            cv2.circle(img, (lx, ly), lr, 0.95, -1)
            cv2.circle(mask, (lx, ly), lr, 1.0, -1)

        # 3. Add Rician-like noise to image
        noise = np.random.normal(0, 0.04, (img_size, img_size))
        img = np.clip(img + noise, 0.0, 1.0)

        images.append(img)
        masks.append(mask)

    np.save(DATA_DIR / "mri_slices.npy", np.array(images, dtype=np.float32))
    np.save(DATA_DIR / "lesion_masks.npy", np.array(masks, dtype=np.float32))
    print(f"[*] Generated {num_samples} synthetic MRI slices and masks at {DATA_DIR}")


if __name__ == "__main__":
    generate_synthetic_mri()
