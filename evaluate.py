"""
Evaluation metrics: Dice Similarity Coefficient (F1) and Intersection over Union (IoU/Jaccard).
"""
import numpy as np


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-5) -> float:
    """Computes Dice Similarity Coefficient between ground truth and predicted binary masks."""
    y_true_f = (y_true > 0.5).astype(np.float32).flatten()
    y_pred_f = (y_pred > 0.5).astype(np.float32).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return float((2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-5) -> float:
    """Computes Intersection over Union (IoU) metric."""
    y_true_f = (y_true > 0.5).astype(np.float32).flatten()
    y_pred_f = (y_pred > 0.5).astype(np.float32).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return float((intersection + smooth) / (union + smooth))
