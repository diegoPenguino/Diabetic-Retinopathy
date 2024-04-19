import torchvision
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

from constants import INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE, UPDATES

from utils import accuracy_fn, accuracy_sickness

loss_fn = nn.CrossEntropyLoss()


class Model_Retinopathy(nn.Module):
    def __init__(self):
        super(Model_Retinopathy, self).__init__()
        self.model = torchvision.models.resnet18(weights="DEFAULT")
        # for param in model.parameters():
        #  param.requires_grad = False
        # It seems like no need to freeze some layers, although i might consider it in the future.
        n_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5),
        )

        self.epochs_trained = 0
        self.losses = []
        self.marks = [0]
        self.accuracies = []
        self.total_accuracies = [0]
        self.validation_accuracies = [0]

    def forward(self, x):
        x = self.model(x)
        return x

    def plot_loss(self):
        plt.axvline(
            x=self.marks[0], color="gray", linestyle="--", label="Epochs"
        )  # This way we put only one label to all epoch marks
        for epoch_mark in self.marks[1:]:
            plt.axvline(x=epoch_mark, color="gray", linestyle="--")
        plt.plot(self.losses, label="Loss")
        plt.plot(get_smooth_losses(self.losses), color="red", label="Loss avg")

        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.accuracies, color="gray")
        plt.plot(self.marks, self.total_accuracies, "o-r")
        plt.plot(self.marks, self.validation_accuracies, "o-b")
        plt.title("Accuracy over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(["Train Accuracy", "Total Accuracy", "Validation Accuracy"])
        plt.show()

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return self.state_dict()

    def get_loss(self, data_loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
                outputs = self(inputs)
                loss += loss_fn(outputs, y_true).item()
        return loss / len(data_loader)

    def get_accuracy(self, data_loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        correct_pred = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
                outputs = self(inputs)
                y_pred = torch.argmax(outputs, dim=1)
                correct_pred += torch.sum(y_pred == y_true).item()
        return correct_pred / len(data_loader.dataset)


def get_smooth_losses(losses: list):
    smooth_losses = []
    for i in range(1, len(losses) + 1):
        smooth_losses.append(np.mean(losses[max(0, i - 50) : i]))
    return smooth_losses


def train_step(model, data_loader, optimizer, update_every=4, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    update_every = len(data_loader) // UPDATES

    model.train()
    correct_pred = 0
    if verbose:
        print(f"Epoch: {model.epochs_trained}")
    for i, batch in enumerate(data_loader):
        inputs, y_true = batch["image"].to(device), batch["labels"].to(device)

        # optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, y_true)
        loss.backward()
        optimizer.step()

        # metrics
        loss = loss.item()
        y_pred = torch.argmax(outputs, dim=1)
        accuracy = accuracy_fn(y_pred, y_true)
        correct_pred += torch.sum(y_pred == y_true).item()
        model.losses.append(loss)
        model.accuracies.append(accuracy)
        acc_sickness = accuracy_sickness(y_pred, y_true)
        if verbose:
            print(
                f"Batch #{i}: Loss: {loss: .5f} Accuracy: {accuracy: .4f} Accuracy sickness: {acc_sickness: .4f}",
                end="\r",
            )

            if i % update_every == 0:
                print(f"Batch #{i}: Loss: {loss: .5f} Accuracy: {accuracy: .4f}")

    model.total_accuracies.append(correct_pred / len(data_loader.dataset))
    model.marks.append(len(model.losses))
    if verbose:
        print("")
        print(
            f"RESULTS - Accuracy: {correct_pred / len(data_loader.dataset): .4f}",
            end=" ",
        )


def validate(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    correct_pred = 0
    avg_loss = 0
    batches = 0
    with torch.inference_mode():
        for batch in data_loader:
            batches += 1
            inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, y_true)
            y_pred = torch.argmax(outputs, dim=1)
            correct_pred += torch.sum(y_pred == y_true).item()
            avg_loss += loss.item()
        avg_loss /= batches
        val_accuracy = correct_pred / len(data_loader.dataset)

        print(f"Validation Accuracy: {val_accuracy: .4f}")
        print(f"Validation Loss: {avg_loss: .4f}")
        model.validation_accuracies.append(val_accuracy)


def train_loop(
    model, train_loader, val_loader, epochs=10, lr=LEARNING_RATE, verbose=True
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_step(model, train_loader, optimizer, verbose=verbose)
        validate(model, val_loader)
        model.epochs_trained += 1
    if verbose:
        print("Training Finished")
