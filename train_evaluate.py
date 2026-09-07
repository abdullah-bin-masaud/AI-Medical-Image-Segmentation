"""
Medical Image Segmentation Training and Evaluation Pipeline.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data.synthetic_medical_generator import generate_synthetic_mri
from models.unet import UNet
from evaluate import dice_coefficient, iou_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "visualizations"


def run_pipeline():
    print("==================================================================")
    print("  AI-BASED MEDICAL IMAGE SEGMENTATION SYSTEM (U-NET)")
    print("==================================================================")

    # 1. Ensure data exists
    if not (DATA_DIR / "mri_slices.npy").exists():
        generate_synthetic_mri(num_samples=50)

    images = np.load(DATA_DIR / "mri_slices.npy")
    masks = np.load(DATA_DIR / "lesion_masks.npy")

    print(f"[*] Loaded dataset: {images.shape[0]} slices ({images.shape[1]}x{images.shape[2]})")

    if torch is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Training on device: {device}")

        # Split 80/20 train/val
        n_train = int(len(images) * 0.8)
        x_train = torch.tensor(images[:n_train]).unsqueeze(1).to(device)
        y_train = torch.tensor(masks[:n_train]).unsqueeze(1).to(device)
        x_val = torch.tensor(images[n_train:]).unsqueeze(1).to(device)
        y_val = torch.tensor(masks[n_train:]).unsqueeze(1).to(device)

        model = UNet(in_channels=1, num_classes=1).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        print("[*] Training U-Net model for 15 epochs...")
        for epoch in range(1, 16):
            model.train()
            optimizer.zero_grad()
            preds = model(x_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch [{epoch:02d}/15] - BCE Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(x_val).cpu().numpy().squeeze()
            val_true = y_val.cpu().numpy().squeeze()

        dice_scores = [dice_coefficient(val_true[i], val_preds[i]) for i in range(len(val_true))]
        iou_scores = [iou_score(val_true[i], val_preds[i]) for i in range(len(val_true))]

        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        print("\n" + "=" * 50)
        print(f"  VALIDATION PERFORMANCE METRICS")
        print(f"  Mean Dice Similarity: {mean_dice:.4f}")
        print(f"  Mean IoU (Jaccard) : {mean_iou:.4f}")
        print("=" * 50)

        # Plot qualitative results
        if plt is not None:
            plt.style.use("dark_background")
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            for i in range(min(3, len(val_preds))):
                axes[i, 0].imshow(images[n_train + i], cmap="gray")
                axes[i, 0].set_title(f"Raw MRI #{i+1}")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(val_true[i], cmap="magma")
                axes[i, 1].set_title("Ground Truth Mask")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(val_preds[i] > 0.5, cmap="cyan" if hasattr(plt.cm, "cyan") else "viridis")
                axes[i, 2].set_title(f"Predicted (Dice={dice_scores[i]:.2f})")
                axes[i, 2].axis("off")

            plt.tight_layout()
            out_plot = PLOTS_DIR / "segmentation_results.png"
            plt.savefig(out_plot, dpi=150)
            plt.close()
            print(f"[*] Visualized segmentation results saved to: {out_plot}")
    else:
        print("[!] PyTorch not installed. Run 'pip install -r requirements.txt' to execute neural network training.")


if __name__ == "__main__":
    run_pipeline()
