from sklearn.metrics import confusion_matrix
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from PIL import Image
from torch import nn
import torchvision
import seaborn as sns
import random

To_Tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

from src.constants import INPUT_SHAPE, BATCH_SIZE

datagen = v2.Compose(
    [
        v2.ToPILImage(),
        v2.Resize(INPUT_SHAPE[:2]),
        # v2.RandomHorizontalFlip(),
        # v2.RandomRotation(20),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ]
)

data_class = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}


class Dataset_Diabetic(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, ext: str, dir: str, augmentation=False):
        self.data = df
        self.data_augmentation = augmentation
        self.ext = ext
        self.dir = dir

    def __len__(self):
        return len(self.data)

    def get_class_distribution(self, idx):
        return data_class[self.data.loc[idx, "level"]]

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.dir,
            self.get_class_distribution(idx),
            self.data.loc[idx, "image"] + "." + self.ext,
        )

        image = Image.open(img_name)
        image = image.resize(INPUT_SHAPE[:2], resample=Image.BILINEAR)
        label = torch.tensor(self.data.loc[idx, "level"])
        image = To_Tensor(image)

        if self.data_augmentation:
            image = datagen(image)
        return {"image": image, "labels": label}


def accuracy_fn(y_pred, y_true) -> float:
    return torch.sum(y_pred == y_true).item() / len(y_true)


def correct_sickness(y_pred, y_true) -> int:
    correct_healthy = torch.sum((y_true == 0) & (y_pred == 0))
    correct_unhealthy = torch.sum((y_true != 0) & (y_pred != 0))
    return correct_healthy + correct_unhealthy


def accuracy_sickness(tp, tn, fp, fn) -> float:
    return (tp + tn) / (tp + tn + fp + fn)


def recall(tp, fn) -> float:
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def precision(tp, fp) -> float:
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def f1_score(precision, recall) -> float:
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def plot_samples(data: Dataset_Diabetic, height: int, width: int):
    fig, ax = plt.subplots(height, width, figsize=(width * 3, height * 3))

    for i in range(height):
        for j in range(width):
            idx = np.random.randint(0, len(data))
            sample = data[idx]
            ax[i, j].imshow(sample["image"].permute(1, 2, 0))
            ax[i, j].set_title(data.get_class_distribution(idx))
            ax[i, j].axis("off")
    plt.show()


def get_dataloader(
    df: pd.DataFrame, ext: str, dir: str, batch_size: int, augmentation=False
):
    dataset = Dataset_Diabetic(df, ext, dir, augmentation)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataset, dataloader


def plot_confusion_matrix(cm: np.ndarray, classes: list):
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(df_cm, annot=True, fmt=".3f", cmap="Blues")
    ax.set(xlabel="Predicted Label", ylabel="True Label")
    plt.show()


def calculate_confusion_matrix(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    confusion_matrix = np.zeros((5, 5), dtype=float)
    with torch.inference_mode():
        for data in data_loader:
            images, labels = data["image"].to(device), data["labels"].to(device)
            outputs = model(images)
            for i in range(len(labels)):
                confusion_matrix[labels[i], torch.argmax(outputs[i])] += 1

    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]
    return confusion_matrix


def calculate_confusion_binary(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    confusion_matrix = np.zeros((2, 2), dtype=float)
    with torch.inference_mode():
        for data in data_loader:
            images, true_y = data["image"].to(device), data["labels"].to(device)
            outputs = model(images)
            pred_y = torch.argmax(outputs, dim=1)
            pred_y = (pred_y > 0).int()
            true_y = (true_y > 0).int()
            for i in range(len(true_y)):
                confusion_matrix[true_y[i], pred_y[i]] += 1

    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]
    return confusion_matrix


def get_accuracies(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode():
        total = 0
        correct = 0
        sickness_correct = 0
        for data in data_loader:
            images, y_true = data["image"].to(device), data["labels"].to(device)
            outputs = model(images)

            y_pred = torch.argmax(outputs, dim=1)
            total += len(y_true)
            correct += torch.sum(y_pred == y_true).item()
            sickness_correct += correct_sickness(y_pred, y_true)
    return correct / total, sickness_correct / total


def sample_iid(df, frac, random_state=None, categories=(0, 1, 2, 3, 4)):
    df_extra = []
    for cat in categories:
        df_sampled = df[df["level"] == cat]
        if random_state is None:
            df_sampled = df_sampled.sample(frac=frac)
        else:
            df_sampled = df_sampled.sample(frac=frac, random_state=random_state)
        df_extra.append(df_sampled)
    return pd.concat(df_extra)


def split_val(df, frac=0.05):
    val_df = sample_iid(df, frac)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def split_for_federated(df, n_clients):
    samples = df.shape[0] // n_clients
    train_df = []

    while len(train_df) < n_clients - 1:
        df_sample = df.sample(n=samples)
        df.drop(df_sample.index, inplace=True)
        train_df.append(df_sample.reset_index(drop=True))
    train_df.append(df.reset_index(drop=True))

    return train_df


## subsplits = 10, 2, 5, 1, 2
def split_non_iid(df, categories=(0, 1, 2, 3, 4), subsplits=(20, 4, 10, 2, 4)):
    healthy, sick = [], []
    for sub, cat in zip(subsplits, categories):
        df_sampled = df[df["level"] == cat].copy()
        aux = split_for_federated(df_sampled, sub)
        if cat == 0:
            healthy.extend(aux)
        else:
            sick.extend(aux)
    assert len(healthy) == len(sick)
    random.shuffle(healthy)
    random.shuffle(sick)
    train_df = []
    for h, s in zip(healthy, sick):
        train_df.append(pd.concat([h, s]).reset_index(drop=True))
    assert len(train_df) == len(sick)
    return train_df


def get_loaders_fed(train_df, ext, directory, batch_size):
    datasets, loaders = [], []
    for small_df in train_df:
        dataset, loader = get_dataloader(small_df, ext, directory, batch_size)
        datasets.append(dataset)
        loaders.append(loader)
    return datasets, loaders
