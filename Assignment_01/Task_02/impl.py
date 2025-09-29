#!/usr/bin/env python3
"""
Assignment 1_2 — Linear Classifiers (SVM & Softmax) on BloodMNIST

Features:
- Vectorized SVM and Softmax losses with gradients (no bias regularization).
- LinearClassifier with SGD mini-batch training and prediction.
- BloodMNIST data loading, preprocessing (flatten, mean-center, bias trick).
- Single-run training and simple hyperparameter search.
- Plots: training loss and optional validation accuracy over iterations.
- Visualization of learned weights as images.
- Evaluation: accuracy, confusion matrix, per-class accuracy.
- CLI for configuration.

Dependencies:
    pip install medmnist matplotlib numpy
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import medmnist
from medmnist import BloodMNIST


# -----------------------------
# Reproducibility
# -----------------------------

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


# -----------------------------
# Utility helpers
# -----------------------------

def _ensure_parent_dir(out_path: str) -> None:
    """Create parent directory for out_path if it exists and is non-empty."""
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute a confusion matrix of shape (C, C). Rows=true class, cols=predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Per-class accuracy from confusion matrix."""
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1, keepdims=False)
        acc[np.isnan(acc)] = 0.0
    return acc


# -----------------------------
# Losses (no bias regularization)
# -----------------------------

def _apply_regularization_no_bias(W: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    L2-regularization excluding the last row (assumed bias).
    Returns (reg_loss, reg_grad).
    """
    if reg <= 0:
        return 0.0, np.zeros_like(W)
    Wnob = W[:-1, :]  # exclude bias row
    reg_loss = reg * np.sum(Wnob * Wnob)
    reg_grad = np.zeros_like(W)
    reg_grad[:-1, :] = 2.0 * reg * Wnob
    return reg_loss, reg_grad


def svm_loss(X: np.ndarray, y: np.ndarray, W: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    Structured SVM (hinge) loss and gradient.
    X: (N, D), y: (N,), W: (D, C)
    """
    if y.ndim > 1:
        y = y.reshape(-1)
    y = y.astype(int)

    N = X.shape[0]
    scores = X.dot(W)                            # (N, C)
    correct = scores[np.arange(N), y][:, None]   # (N, 1)
    margins = np.maximum(0.0, scores - correct + 1.0)
    margins[np.arange(N), y] = 0.0

    loss = np.sum(margins) / N
    reg_loss, reg_grad = _apply_regularization_no_bias(W, reg)
    loss += reg_loss

    binary = (margins > 0).astype(scores.dtype)
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(N), y] -= row_sum
    dW = (X.T.dot(binary)) / N
    dW += reg_grad
    return loss, dW


def softmax_loss(X: np.ndarray, y: np.ndarray, W: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    Softmax cross-entropy loss and gradient (with numerical stability).
    X: (N, D), y: (N,), W: (D, C)
    """
    if y.ndim > 1:
        y = y.reshape(-1)
    y = y.astype(int)

    N = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)  # stability

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-12)

    loss = np.mean(correct_logprobs)
    reg_loss, reg_grad = _apply_regularization_no_bias(W, reg)
    loss += reg_loss

    dscores = probs
    dscores[np.arange(N), y] -= 1.0
    dscores /= N
    dW = X.T.dot(dscores)
    dW += reg_grad
    return loss, dW


# -----------------------------
# Classifier
# -----------------------------

class LinearClassifier:
    """
    Linear classifier trained with mini-batch SGD.
    Supports 'svm' and 'softmax' losses.
    """

    def __init__(self, input_dim: int, num_classes: int, loss_function: str = "svm", reg: float = 0.0):
        self.W = 0.001 * np.random.randn(input_dim, num_classes)
        self.loss_function = loss_function
        self.reg = float(reg)

    def _loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.loss_function == "svm":
            return svm_loss(X, y, self.W, self.reg)
        elif self.loss_function == "softmax":
            return softmax_loss(X, y, self.W, self.reg)
        raise ValueError(f"Invalid loss function: {self.loss_function}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 1e-3,
        num_iters: int = 1000,
        batch_size: int = 200,
        verbose: bool = False,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        eval_interval: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Train with SGD mini-batches. Optionally track validation accuracy.
        Returns a history dict: {"loss": [...], "val_acc": [...]}
        """
        N = X.shape[0]
        history: Dict[str, List[float]] = {"loss": []}
        if X_val is not None and y_val is not None:
            history["val_acc"] = []

        for it in range(num_iters):
            idx = np.random.choice(N, size=batch_size, replace=True)
            Xb = X[idx]
            yb = y[idx]

            loss, grad = self._loss_and_grad(Xb, yb)
            history["loss"].append(float(loss))

            self.W -= learning_rate * grad

            if verbose and (it % 100 == 0):
                print(f"Iteration {it}/{num_iters}  loss={loss:.4f}")

            if (X_val is not None) and (y_val is not None) and (eval_interval > 0) and (it % eval_interval == 0):
                val_pred = self.predict(X_val)
                val_acc = float(np.mean(val_pred == y_val))
                history["val_acc"].append(val_acc)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X.dot(self.W)
        return np.argmax(scores, axis=1)


# -----------------------------
# Data pipeline
# -----------------------------

def load_bloodmnist(size: int = 28) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load BloodMNIST train/val/test splits.
    Returns (Xtr, ytr, Xval, yval, Xte, yte, label_info)
    """
    train_ds = BloodMNIST(split="train", download=True, size=size)
    val_ds   = BloodMNIST(split="val",   download=True, size=size)
    test_ds  = BloodMNIST(split="test",  download=True, size=size)

    return (
        train_ds.imgs, train_ds.labels,
        val_ds.imgs,   val_ds.labels,
        test_ds.imgs,  test_ds.labels,
        train_ds.info["label"],
    )


def preprocess_flatten_normalize_add_bias(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten images, convert to float32, subtract train-mean image, append bias column of 1s.
    """
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_val   = X_val.reshape(X_val.shape[0], -1).astype(np.float32)
    X_test  = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val   -= mean_image
    X_test  -= mean_image

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)])
    X_val   = np.hstack([X_val,   np.ones((X_val.shape[0],   1), dtype=X_val.dtype)])
    X_test  = np.hstack([X_test,  np.ones((X_test.shape[0],  1), dtype=X_test.dtype)])

    return X_train, X_val, X_test


def to_int_labels(y: np.ndarray) -> np.ndarray:
    """BloodMNIST labels come as shape (N,1); flatten and cast to int."""
    return y.reshape(-1).astype(int)


# -----------------------------
# Training / search / evaluation
# -----------------------------

def compute_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(y_pred == y_true))


def train_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_type: str = "svm",
    reg: float = 0.0,
    learning_rate: float = 1e-3,
    num_iters: int = 1000,
    batch_size: int = 200,
    verbose: bool = False,
    eval_interval: int = 100,
) -> Tuple[LinearClassifier, Dict[str, List[float]], float]:
    input_dim = X_train.shape[1]
    num_classes = int(np.unique(y_train).size)
    clf = LinearClassifier(input_dim=input_dim, num_classes=num_classes, loss_function=loss_type, reg=reg)

    history = clf.train(
        X_train,
        y_train,
        learning_rate=learning_rate,
        num_iters=num_iters,
        batch_size=batch_size,
        verbose=verbose,
        X_val=X_val,
        y_val=y_val,
        eval_interval=eval_interval,
    )
    val_acc = compute_accuracy(clf.predict(X_val), y_val)
    return clf, history, val_acc


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_type: str = "svm",
    learning_rates: List[float] | None = None,
    regularizations: List[float] | None = None,
    num_iters: int = 1000,
    batch_size: int = 200,
    verbose: bool = False,
) -> Tuple[LinearClassifier, Dict[str, float], List[Dict]]:
    if learning_rates is None:
        learning_rates = [1e-2, 5e-3, 1e-3, 5e-4]
    if regularizations is None:
        regularizations = [1e-3, 1e-2, 1e-1, 1.0]

    best_model = None
    best_acc = -1.0
    best_params: Dict[str, float] = {}
    results: List[Dict] = []

    for lr in learning_rates:
        for reg in regularizations:
            clf, hist, val_acc = train_once(
                X_train, y_train, X_val, y_val,
                loss_type=loss_type, reg=reg, learning_rate=lr,
                num_iters=num_iters, batch_size=batch_size, verbose=verbose
            )
            results.append({"lr": lr, "reg": reg, "val_acc": val_acc, "history": hist})
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = clf
                best_params = {"lr": lr, "reg": reg}

    assert best_model is not None
    return best_model, best_params, results


# -----------------------------
# Visualization
# -----------------------------

def visualize_weights(classifier: LinearClassifier, out_path: str, image_size: Tuple[int, int, int] = (28, 28, 3)) -> None:
    """
    Save a figure showing one weight image per class.
    """
    _ensure_parent_dir(out_path)
    W = classifier.W[:-1, :]  # drop bias row
    h, w, c = image_size
    C = W.shape[1]
    W = W.reshape(h, w, c, C)

    w_min, w_max = float(np.min(W)), float(np.max(W))
    plt.figure(figsize=(2 * C, 2))
    for i in range(C):
        plt.subplot(1, C, i + 1)
        wimg = (W[:, :, :, i] - w_min) / (w_max - w_min + 1e-12)
        wimg = (255.0 * wimg).astype("uint8")
        plt.imshow(wimg)
        plt.axis("off")
        plt.title(str(i))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_history(history: Dict[str, List[float]], out_prefix: str) -> None:
    """
    Save training loss curve and, if present, validation accuracy curve.
    """
    _ensure_parent_dir(out_prefix)

    # Loss plot
    loss_path = f"{out_prefix}_loss.png"
    plt.figure()
    plt.plot(history.get("loss", []))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # Val acc plot (sampled every eval_interval)
    if "val_acc" in history and len(history["val_acc"]) > 0:
        acc_path = f"{out_prefix}_valacc.png"
        plt.figure()
        plt.plot(history["val_acc"])
        plt.xlabel("Eval step (every eval_interval)")
        plt.ylabel("Validation accuracy")
        plt.title("Validation Accuracy (periodic)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(acc_path)
        plt.close()


# -----------------------------
# Main / CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Linear classifier on BloodMNIST")
    parser.add_argument("--loss", choices=["svm", "softmax"], default="svm")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-2)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate val accuracy every N iterations (0 to disable)")
    parser.add_argument("--size", type=int, default=28, help="BloodMNIST resize (28 default)")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory for saved plots")
    args = parser.parse_args()

    # Load data
    Xtr_img, ytr_raw, Xval_img, yval_raw, Xte_img, yte_raw, label_info = load_bloodmnist(size=args.size)

    # Preprocess
    X_train, X_val, X_test = preprocess_flatten_normalize_add_bias(Xtr_img, Xval_img, Xte_img)
    y_train = to_int_labels(ytr_raw)
    y_val   = to_int_labels(yval_raw)
    y_test  = to_int_labels(yte_raw)

    # Ensure plots dir exists
    os.makedirs(args.plots_dir, exist_ok=True)

    if args.search:
        best_model, best_params, results = hyperparameter_search(
            X_train, y_train, X_val, y_val,
            loss_type=args.loss,
            learning_rates=[1e-2, 5e-3, 1e-3, 5e-4],
            regularizations=[1e-3, 1e-2, 1e-1, 1.0],
            num_iters=args.iters,
            batch_size=args.batch_size,
            verbose=False,
        )
        print(f"Best {args.loss} params: {best_params}")

        # Pick best run and plot its history
        best_idx = int(np.argmax([r["val_acc"] for r in results]))
        plot_history(results[best_idx]["history"], os.path.join(args.plots_dir, f"history_best_{args.loss}"))

        # Test evaluation
        y_test_pred = best_model.predict(X_test)
        test_acc = compute_accuracy(y_test_pred, y_test)
        print(f"Test accuracy: {test_acc:.4f}")

        # Confusion matrix & per-class acc
        C = int(np.unique(y_train).size)
        cm = confusion_matrix(y_test, y_test_pred, C)
        pca = per_class_accuracy(cm)
        print("Per-class accuracy (test):", np.round(pca, 4))
        print("Confusion matrix (rows=true, cols=pred):\n", cm)

        # Weights visualization
        visualize_weights(best_model, os.path.join(args.plots_dir, f"weights_{args.loss}.png"), image_size=(args.size, args.size, 3))

    else:
        model, history, val_acc = train_once(
            X_train, y_train, X_val, y_val,
            loss_type=args.loss, reg=args.reg, learning_rate=args.lr,
            num_iters=args.iters, batch_size=args.batch_size, verbose=True,
            eval_interval=args.eval_interval,
        )
        print(f"Val accuracy: {val_acc:.4f}")

        # Plots
        plot_history(history, os.path.join(args.plots_dir, f"history_{args.loss}"))

        # Test evaluation
        y_test_pred = model.predict(X_test)
        test_acc = compute_accuracy(y_test_pred, y_test)
        print(f"Test accuracy: {test_acc:.4f}")

        # Confusion matrix & per-class acc
        C = int(np.unique(y_train).size)
        cm = confusion_matrix(y_test, y_test_pred, C)
        pca = per_class_accuracy(cm)
        print("Per-class accuracy (test):", np.round(pca, 4))
        print("Confusion matrix (rows=true, cols=pred):\n", cm)

        # Weights visualization
        visualize_weights(model, os.path.join(args.plots_dir, f"weights_single_run_{args.loss}.png"), image_size=(args.size, args.size, 3))


if __name__ == "__main__":
    main()
