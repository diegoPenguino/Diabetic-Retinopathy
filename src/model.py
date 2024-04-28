import copy
import torchvision
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.constants import INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE, UPDATES
from src.constants import ext, directory

from src.utils import accuracy_fn, accuracy_sickness, get_dataloader, split_val

loss_fn = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def get_resnet18():
    model = torchvision.models.resnet18(weights="DEFAULT")
    freeze_model(model)

    n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 5),
    )
    return model


def get_vgg19():
    model = torchvision.models.vgg19(weights="DEFAULT")
    freeze_model(model)
    model.classifier.add_module("final", nn.Linear(1000, 5))
    return model


class Model_Retinopathy(nn.Module):
    def __init__(self, optimizer_fn, train_df, val_loader, lr=LEARNING_RATE):
        super(Model_Retinopathy, self).__init__()
        self.model = get_resnet18()
        for param in self.model.parameters():
            param.requires_grad = True

        ##FIX when using parameters(), to only use the ones which have requires_grad = True
        self.lr = lr
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)
        self.epochs_trained = 0
        _, self.train_loader = get_dataloader(train_df, ext, directory, BATCH_SIZE)
        self.val_loader = val_loader

        self.losses = []
        self.accuracies = []
        self.total_accuracies = [0]

        self.marks = [0]
        self.val_accuracies = [0]
        self.val_losses = []

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def forward(self, x):
        x = self.model(x)
        return x

    def update_lr(self, new_lr):
        self.lr = new_lr
        self.optimizer = self.optimizer_fn(self.parameters(), lr=self.lr)

    def unfreeze(self, percentage):
        for i, param in enumerate(self.parameters()):
            condition = i > len(list(self.parameters())) * (1 - percentage)
            param.requires_grad = condition

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.axvline(
            x=self.marks[0], color="gray", linestyle="--", label="Epochs Marks"
        )  # This way we put only one label to all epoch marks
        for epoch_mark in self.marks[1:]:
            ax.axvline(x=epoch_mark, color="gray", linestyle="--")
        ax.plot(self.losses, label="Loss")
        ax.plot(get_smooth_losses(self.losses), color="red", label="Loss avg")

        ax.set_title("Loss over epochs")
        ax.set_xlabel("Batches")
        ax.set_ylabel("Loss")
        ax.legend()
        return ax

    def plot_accuracy(self):
        fig, ax = plt.subplots()
        ax.plot(self.accuracies, color="gray")
        ax.plot(self.marks, self.total_accuracies, "o-r")
        ax.plot(self.marks, self.validation_accuracies, "o-b")
        ax.set_title("Accuracy over epochs")
        ax.set_xlabel("Batches")
        ax.set_ylabel("Accuracy")
        ax.legend(["Train Accuracy", "Total Accuracy", "Validation Accuracy"])
        return ax

    def save_plots(self):  ## MOVE TO server
        plt.ioff()

        ax = self.plot_loss()
        plt.savefig("Losses.png")
        plt.close()

        ax = self.plot_accuracy()
        plt.savefig("Accuracies.png")
        plt.close()

    def plot_val_loss(self, ax):
        ax.plot(self.marks, self.val_losses, linestyle="-", color="red")

    def plot_val_accuracy(self, ax):
        ax.plot(self.marks, self.val_accuracies, linestyle="-", color="red")

    def check_loader(self, loader):
        if loader is None:
            loader = self.val_loader
        if loader == "train":
            loader = self.train_loader
        if loader == "val":
            loader = self.val_loader
        return loader

    def get_loss(self, loader=None):
        loader = self.check_loader(loader)

        self.model.eval()
        loss = 0
        with torch.inference_mode():
            for batch in loader:
                inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
                outputs = self(inputs)
                loss += loss_fn(outputs, y_true).item()
        return loss / len(loader)

    def get_accuracy(self, loader=None):
        loader = self.check_loader(loader)
        self.model.eval()
        correct_pred = 0
        with torch.inference_mode():
            for batch in loader:
                inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
                outputs = self(inputs)
                y_pred = torch.argmax(outputs, dim=1)
                correct_pred += torch.sum(y_pred == y_true).item()
        return correct_pred / len(loader.dataset)

    def validate(self, loader=None):
        val_loss = self.get_loss(loader)
        val_accuracy = self.get_accuracy(loader)
        return {"accuracy": val_accuracy, "loss": val_loss}

    def train_step(self, update_every=4, verbose=True):
        update_every = len(self.train_loader) // UPDATES

        self.model.train()
        correct_pred = 0
        if verbose:
            print(f"Epoch: {self.epochs_trained}")
        for i, batch in enumerate(self.train_loader):
            inputs, y_true = batch["image"].to(device), batch["labels"].to(device)

            # optimizer step
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = loss_fn(outputs, y_true)
            loss.backward()
            self.optimizer.step()

            # metrics
            loss = loss.item()
            y_pred = torch.argmax(outputs, dim=1)
            accuracy = accuracy_fn(y_pred, y_true)
            correct_pred += torch.sum(y_pred == y_true).item()

            self.losses.append(loss)
            self.accuracies.append(accuracy)
            if verbose is None:
                print(
                    f"Batch #{i}: Loss: {loss: .5f} Accuracy: {accuracy: .4f}",
                    end="\r",
                )

                if i % update_every == 0:
                    print(f"Batch #{i}: Loss: {loss: .5f} Accuracy: {accuracy: .4f}")

        self.total_accuracies.append(correct_pred / len(self.train_loader.dataset))
        self.epochs_trained += 1
        self.marks.append(self.epochs_trained)

    def train_loop(self, epochs, verbose=True):
        if self.epochs_trained == 0:
            self.val_losses.append(self.get_loss())
        for _ in range(epochs):
            self.train_step(verbose=verbose)
            val = self.validate()
            acc = val["accuracy"]
            self.append_val_metrics(val)
            # self.save_plots()

            if verbose:
                print("VALIDATION: ", val)
        if verbose:
            print("Training Finished")

    def append_val_metrics(self, val):
        self.val_losses.append(val["loss"])
        self.val_accuracies.append(val["accuracy"])

    def get_data_len(self):
        return len(self.train_loader.dataset)


def get_smooth_losses(losses: list):
    smooth_losses = []
    for i in range(1, len(losses) + 1):
        smooth_losses.append(np.mean(losses[max(0, i - 50) : i]))
    return smooth_losses
