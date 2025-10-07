#!/usr/bin/env python3
"""
impl.py

Runnable script to train, tune, visualize and evaluate a simple fully-connected
neural network on the BloodMNIST dataset (from medmnist).

This implements the three previously-empty notebook sections:
 - Your optimization code goes here
 - Visualize hyperparameter tuning history/results
 - evaluate on the test data (includes confusion matrix + classification report)

Usage from terminal:
    python impl.py --epochs 50 --batch_size 128

Outputs:
 - Saved plots in ./Plots/ (validation histories, hyperparameter search plot, confusion matrix)
 - Model and history saved as numpy files (optional)

The code intentionally keeps dependencies minimal (numpy, matplotlib, scikit-learn, medmnist).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Try to import medmnist and sklearn; install if missing
import importlib
import subprocess
import sys

reqs = [("medmnist", "medmnist"), ("sklearn", "scikit-learn")]
for pkg, pip_name in reqs:
    if importlib.util.find_spec(pkg) is None:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

from medmnist import BloodMNIST
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ----------------------------- Model ---------------------------------
class FullyConnectedNN:
    def __init__(self, layer_sizes, reg=1e-3, seed=42, loss='softmax'):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes)
        self.reg = reg
        self.loss = loss
        self.params = {}
        for i in range(1, self.L):
            self.params[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2.0/layer_sizes[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_sizes[i]))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_der(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        cache = {'A0': X}
        A = X
        for i in range(1, self.L):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            Z = A.dot(W) + b
            cache[f'Z{i}'] = Z
            if i < self.L-1:
                A = self.relu(Z)
            else:
                A = Z
            cache[f'A{i}'] = A
        return A, cache

    def compute_loss(self, scores, y):
        m = y.shape[0]
        probs = self.softmax(scores)
        correct_logprobs = -np.log(probs[np.arange(m), y] + 1e-12)
        data_loss = np.sum(correct_logprobs)/m
        reg_loss = 0
        for i in range(1, self.L):
            reg_loss += 0.5 * self.reg * np.sum(self.params[f'W{i}']**2)
        return data_loss + reg_loss

    def backward(self, cache, y):
        grads = {}
        m = y.shape[0]
        A_last = cache[f'A{self.L-1}']
        probs = self.softmax(A_last)
        probs[np.arange(m), y] -= 1
        dA = probs / m
        for i in range(self.L-1, 0, -1):
            A_prev = cache[f'A{i-1}']
            grads[f'W{i}'] = A_prev.T.dot(dA) + self.reg * self.params[f'W{i}']
            grads[f'b{i}'] = np.sum(dA, axis=0, keepdims=True)
            if i > 1:
                dZ_prev = dA.dot(self.params[f'W{i}'].T)
                dA = dZ_prev * self.relu_der(cache[f'Z{i-1}'])
        return grads

    def update_params(self, grads, lr, opt_state=None, optimizer='sgd', beta1=0.9, beta2=0.999, t=1):
        # opt_state can hold velocities or moments
        if opt_state is None:
            opt_state = {}
        for i in range(1, self.L):
            if optimizer == 'sgd':
                self.params[f'W{i}'] -= lr * grads[f'W{i}']
                self.params[f'b{i}'] -= lr * grads[f'b{i}']
            elif optimizer == 'momentum':
                vW = opt_state.setdefault(f'vW{i}', np.zeros_like(self.params[f'W{i}']))
                vb = opt_state.setdefault(f'vb{i}', np.zeros_like(self.params[f'b{i}']))
                vW[:] = beta1 * vW + (1 - beta1) * grads[f'W{i}']
                vb[:] = beta1 * vb + (1 - beta1) * grads[f'b{i}']
                self.params[f'W{i}'] -= lr * vW
                self.params[f'b{i}'] -= lr * vb
            elif optimizer == 'adam':
                mW = opt_state.setdefault(f'mW{i}', np.zeros_like(self.params[f'W{i}']))
                vW = opt_state.setdefault(f'vW{i}', np.zeros_like(self.params[f'W{i}']))
                mb = opt_state.setdefault(f'mb{i}', np.zeros_like(self.params[f'b{i}']))
                vb = opt_state.setdefault(f'vb{i}', np.zeros_like(self.params[f'b{i}']))
                mW[:] = beta1*mW + (1-beta1)*grads[f'W{i}']
                mb[:] = beta1*mb + (1-beta1)*grads[f'b{i}']
                vW[:] = beta2*vW + (1-beta2)*(grads[f'W{i}']**2)
                vb[:] = beta2*vb + (1-beta2)*(grads[f'b{i}']**2)
                mW_hat = mW / (1 - beta1**t)
                mb_hat = mb / (1 - beta1**t)
                vW_hat = vW / (1 - beta2**t)
                vb_hat = vb / (1 - beta2**t)
                self.params[f'W{i}'] -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
                self.params[f'b{i}'] -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
            else:
                raise ValueError('Unsupported optimizer')
        return opt_state

    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)


# ----------------------------- Utilities ------------------------------

def prepare_data():
    ds_train = BloodMNIST(split='train', download=True, size=28)
    ds_val = BloodMNIST(split='val', download=True, size=28)
    ds_test = BloodMNIST(split='test', download=True, size=28)

    X_train = ds_train.imgs.astype(np.float32)
    y_train = ds_train.labels.astype(np.int64).flatten()
    X_val = ds_val.imgs.astype(np.float32)
    y_val = ds_val.labels.astype(np.int64).flatten()
    X_test = ds_test.imgs.astype(np.float32)
    y_test = ds_test.labels.astype(np.int64).flatten()

    # Normalize and flatten
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_val = X_val.reshape((X_val.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    return X_train, y_train, X_val, y_val, X_test, y_test


def minibatch_iter(X, y, batch_size=64, shuffle=True):
    m = X.shape[0]
    idx = np.arange(m)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, m, batch_size):
        batch_idx = idx[i:i+batch_size]
        yield X[batch_idx], y[batch_idx]


# -------------------- Optimization code (training + tuning) -------------

def train_model(X_train, y_train, X_val, y_val, config):
    # config: dict with network and training hyperparameters
    layer_sizes = config.get('layer_sizes')
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 128)
    lr = config.get('lr', 1e-3)
    reg = config.get('reg', 1e-3)
    optimizer = config.get('optimizer', 'sgd')

    net = FullyConnectedNN(layer_sizes, reg=reg)
    opt_state = {}
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    t = 1
    for epoch in range(1, epochs+1):
        # training
        losses = []
        for Xb, yb in minibatch_iter(X_train, y_train, batch_size=batch_size, shuffle=True):
            scores, cache = net.forward(Xb)
            loss = net.compute_loss(scores, yb)
            grads = net.backward(cache, yb)
            opt_state = net.update_params(grads, lr, opt_state=opt_state, optimizer=optimizer, t=t)
            losses.append(loss)
            t += 1
        avg_train_loss = float(np.mean(losses))

        # validation
        val_scores, _ = net.forward(X_val)
        val_loss = float(net.compute_loss(val_scores, y_val))
        val_preds = np.argmax(val_scores, axis=1)
        val_acc = float(np.mean(val_preds == y_val))

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % max(1, epochs//10) == 0 or epoch==1:
            print(f"Epoch {epoch}/{epochs} - train_loss: {avg_train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")
    return net, history


def hyperparameter_search(X_train, y_train, X_val, y_val, search_space, base_config, save_plots_dir):
    results = []
    if not os.path.exists(save_plots_dir):
        os.makedirs(save_plots_dir)

    for lr in search_space.get('lr', [1e-3]):
        for reg in search_space.get('reg', [1e-3]):
            for opt in search_space.get('optimizer', ['sgd']):
                cfg = base_config.copy()
                cfg.update({'lr': lr, 'reg': reg, 'optimizer': opt})
                print(f"Training with lr={lr}, reg={reg}, opt={opt}")
                model, history = train_model(X_train, y_train, X_val, y_val, cfg)
                best_val_acc = max(history['val_acc'])
                results.append({'lr': lr, 'reg': reg, 'optimizer': opt, 'history': history, 'best_val_acc': best_val_acc, 'model': model})

    # Visualize hyperparameter tuning history/results
    # We'll save a simple comparison plot (best_val_acc for each configuration)
    labels = [f"lr={r['lr']}, reg={r['reg']}, opt={r['optimizer']}" for r in results]
    bests = [r['best_val_acc'] for r in results]

    plt.figure(figsize=(10,6))
    x = np.arange(len(results))
    plt.bar(x, bests)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Best val acc')
    plt.title('Hyperparameter search results (best validation accuracy)')
    plt.tight_layout()
    fn = os.path.join(save_plots_dir, 'hyperparam_search_best_val_acc.png')
    plt.savefig(fn)
    print(f"Saved hyperparameter summary to {fn}")
    plt.close()

    # Additionally save the validation accuracy curves for top 3 configs
    results_sorted = sorted(results, key=lambda r: r['best_val_acc'], reverse=True)
    for i, r in enumerate(results_sorted[:3]):
        plt.figure()
        plt.plot(r['history']['val_acc'], label='val_acc')
        plt.plot(r['history']['train_loss'], label='train_loss')
        plt.xlabel('Epoch')
        plt.title(f"Top {i+1}: lr={r['lr']}, reg={r['reg']}, opt={r['optimizer']} (best={r['best_val_acc']:.3f})")
        plt.legend()
        plt.tight_layout()
        fn = os.path.join(save_plots_dir, f'top{i+1}_history.png')
        plt.savefig(fn)
        plt.close()
        print(f"Saved history plot to {fn}")

    return results


# ----------------------- Evaluation ----------------------------------

def evaluate_model(model, X_test, y_test, save_plots_dir, class_names=None):
    if not os.path.exists(save_plots_dir):
        os.makedirs(save_plots_dir)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_labels = class_names if class_names is not None else np.arange(cm.shape[0])
    plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45)
    plt.yticks(np.arange(len(tick_labels)), tick_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    fn = os.path.join(save_plots_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()
    print(f"Saved confusion matrix to {fn}")

    # Classification report
    report = classification_report(y_test, preds)
    print("Classification report:\n", report)
    with open(os.path.join(save_plots_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    return acc, cm, report


# ----------------------- Main / CLI ----------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--do_search', action='store_true')
    parser.add_argument('--plots_dir', type=str, default='Plots')
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    class_names = [str(i) for i in range(n_classes)]

    base_config = {
        'layer_sizes': [n_features, 256, 128, n_classes],
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': 1e-3,
        'reg': 1e-3,
        'optimizer': 'adam'
    }

    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    if args.do_search:
        search_space = {'lr': [1e-3, 3e-4], 'reg': [1e-3, 1e-4], 'optimizer': ['sgd', 'adam', 'momentum']}
        results = hyperparameter_search(X_train, y_train, X_val, y_val, search_space, base_config, args.plots_dir)
        # pick best
        best = max(results, key=lambda r: r['best_val_acc'])
        model = best['model']
        # save best history
        np.savez(os.path.join(args.plots_dir, 'hp_search_results.npz'), best_val_acc=best['best_val_acc'])
    else:
        model, history = train_model(X_train, y_train, X_val, y_val, base_config)
        # save one history plot
        plt.figure()
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.plot(history['val_acc'], label='val_acc')
        plt.legend()
        plt.title('Training history')
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, 'training_history.png'))
        plt.close()

    # Evaluate on test set
    evaluate_model(model, X_test, y_test, args.plots_dir, class_names=class_names)


if __name__ == '__main__':
    main()
