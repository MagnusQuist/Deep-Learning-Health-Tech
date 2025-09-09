import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from medmnist import BloodMNIST
import os


class KNearestNeighbor:
    def __init__(self, distance="l2"):
        self.X_train = None
        self.y_train = None
        self.distance = distance

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def calculate_distances(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            if self.distance == "l1":
                dists[i, :] = np.sum(np.abs(self.X_train - X_test[i, :]), axis=1)
            else:  # default l2
                dists[i, :] = np.sqrt(np.sum((self.X_train - X_test[i, :]) ** 2, axis=1))
        return dists

    def predict(self, X_test, k=5):
        dists = self.calculate_distances(X_test)
        num_test = X_test.shape[0]
        y_pred = np.zeros(num_test, dtype=int)
        neighbors_idx = []

        for i in range(num_test):
            neighbors = np.argsort(dists[i])[:k]
            closest_y = self.y_train[neighbors]
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)
            neighbors_idx.append(neighbors)

        return y_pred, neighbors_idx


def load_data():
    train_dataset = BloodMNIST(split="train", download=True, size=28)
    val_dataset = BloodMNIST(split="val", download=True, size=28)

    train_images, train_labels = train_dataset.imgs, train_dataset.labels
    val_images, val_labels = val_dataset.imgs, val_dataset.labels
    label_info = train_dataset.info["label"]

    print("Træningsdata:")
    print(f"Billeder: {train_images.shape}, Labels: {train_labels.shape}")
    print("Valideringsdata:")
    print(f"Billeder: {val_images.shape}, Labels: {val_labels.shape}")

    return train_images, train_labels, val_images, val_labels, label_info


def plot_examples(images, labels, label_info, title, save_path):
    random.seed(42)
    fig, axes = plt.subplots(5, len(label_info), figsize=(15, 5))

    for class_, name in label_info.items():
        class_indices = [idx for idx, label in enumerate(labels) if int(class_) == label]
        selected_indices = random.sample(class_indices, 5)
        for j, idx in enumerate(selected_indices):
            image, label = images[idx], labels[idx]
            axes[j, int(class_)].imshow(image, cmap="gray")
            axes[j, int(class_)].axis("off")
            if j == 0:
                axes[j, int(class_)].set_title(f"{name[:5]}: {class_}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_nearest_neighbors(X_val, y_val, X_train, y_train, neighbors_idx, save_path, num_samples=5):
    fig, axes = plt.subplots(num_samples, 6, figsize=(15, 6))
    selected = np.random.choice(len(X_val), num_samples, replace=False)

    for i, idx in enumerate(selected):
        test_img = X_val[idx].reshape(28, 28, 3)
        axes[i, 0].imshow(test_img)
        axes[i, 0].set_title(f"Val: {y_val[idx]}")
        axes[i, 0].axis("off")

        nn_indices = neighbors_idx[idx][:5]
        for j, nn_idx in enumerate(nn_indices):
            nn_img = X_train[nn_idx].reshape(28, 28, 3)
            axes[i, j + 1].imshow(nn_img)
            axes[i, j + 1].set_title(f"NN {j+1}: {y_train[nn_idx]}")
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def preprocess(train_images, train_labels, val_images, val_labels, num_training=5000, num_val=500):
    np.random.seed(0)

    randomize = np.arange(train_images.shape[0])
    np.random.shuffle(randomize)
    X_train = train_images[randomize]
    y_train = train_labels[randomize].flatten()

    randomize_val = np.arange(val_images.shape[0])
    np.random.shuffle(randomize_val)
    X_val = val_images[randomize_val]
    y_val = val_labels[randomize_val].flatten()

    X_train = X_train[:num_training]
    y_train = y_train[:num_training]
    X_val = X_val[:num_val]
    y_val = y_val[:num_val]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    print(f"New train shape: {X_train.shape}")
    print(f"New val shape: {X_val.shape}")

    return X_train, y_train, X_val, y_val


def hyperparameter_search(knn, X_val, y_val, neighbors_list, save_path):
    results = {}
    for k in neighbors_list:
        print(f"Testing k={k}...")
        predictions, _ = knn.predict(X_val, k=k)
        num_correct = np.sum(predictions == y_val)
        acc = (num_correct / y_val.shape[0]) * 100
        results[k] = acc
        print(f"Accuracy for k={k}: {acc:.2f}%")

    plt.figure()
    plt.plot(list(results.keys()), list(results.values()), marker="o")
    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.title("KNN Hyperparameter Search")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    return results


def test_best_model(knn, X_train, y_train, best_k, distance):
    test_dataset = BloodMNIST(split="test", download=True, size=28)
    X_test, y_test = test_dataset.imgs, test_dataset.labels
    print("Testsdata:")
    print(f"Billeder: {X_test.shape}, Labels: {y_test.shape}")

    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    y_test = y_test.flatten()
    print(X_test.shape, y_test.shape)

    knn.train(X_train, y_train)
    predictions, _ = knn.predict(X_test, k=best_k)
    num_correct = np.sum(predictions == y_test)
    accuracy = num_correct / y_test.shape[0] * 100
    print(f"Final Test Accuracy for k={best_k} with {distance} distance: {accuracy:.2f}%")


def main(k: int, distance: str):
    os.makedirs("plots", exist_ok=True)

    train_images, train_labels, val_images, val_labels, label_info = load_data()
    plot_examples(train_images, train_labels, label_info, "Training Examples", "plots/train_examples.png")
    plot_examples(val_images, val_labels, label_info, "Validation Examples", "plots/val_examples.png")

    X_train, y_train, X_val, y_val = preprocess(train_images, train_labels, val_images, val_labels)

    knn = KNearestNeighbor(distance=distance)
    knn.train(X_train, y_train)

    predictions, neighbors_idx = knn.predict(X_val, k=k)
    num_correct = np.sum(predictions == y_val)
    accuracy = num_correct / y_val.shape[0] * 100
    print(f"\nKNN accuracy for k={k} with {distance} distance: {accuracy:.2f}%")

    plot_nearest_neighbors(X_val, y_val, X_train, y_train, neighbors_idx, "plots/nearest_neighbors.png")

    neighbors = [1, 5, 13, 33, 67, 79, 91, 131, 167, 201]
    results = hyperparameter_search(knn, X_val, y_val, neighbors, "plots/hyperparam_search.png")

    best_k = max(results, key=results.get)
    print(f"Best k based on validation: {best_k} with accuracy {results[best_k]:.2f}%")

    test_best_model(knn, X_train, y_train, best_k, distance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN on BloodMNIST dataset")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--distance", type=str, choices=["l1", "l2"], default="l2", help="Distance metric")
    args = parser.parse_args()

    main(args.k, args.distance)
