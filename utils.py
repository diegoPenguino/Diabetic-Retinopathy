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
from zmq import device

To_Tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

from constants import INPUT_SHAPE, BATCH_SIZE

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
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            correct += 1
        elif y_true[i] != 0 and y_pred[i] != 0:
            correct += 1
    return correct


def accuracy_sickness(y_pred, y_true) -> float:
    return correct_sickness(y_pred, y_true) / len(y_true)


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


def sample_iid(df, frac, random_state=None, categories=[0, 1, 2, 3, 4]):
    df_extra = []
    for cat in categories:
        df_sampled = df[df["level"] == cat]
        if random_state is None:
            df_sampled = df_sampled.sample(frac=frac)
        else:
            df_sampled = df_sampled.sample(frac=frac, random_state=random_state)
        df_extra.append(df_sampled)
    return pd.concat(df_extra)
