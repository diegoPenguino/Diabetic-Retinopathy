import copy
import torchvision
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.constants import INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE, UPDATES

from src.utils import accuracy_fn, accuracy_sickness

loss_fn = nn.CrossEntropyLoss()


class Model_Retinopathy(nn.Module):
    def __init__(self, optimizer_fn=torch.optim.SGD):
        super(Model_Retinopathy, self).__init__()
        # self.model = torchvision.models.resnet18(weights="DEFAULT")
        self.model = torchvision.models.vgg19(weights="DEFAULT")
        for param in self.parameters():
            param.requires_grad = False
        self.model.classifier.add_module("final", nn.Linear(1000, 5))
        """ n_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5),
        ) """
        self.lr = LEARNING_RATE
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.parameters(), lr=self.lr)
        self.epochs_trained = 0
        self.losses = []
        self.marks = [0]
        self.accuracies = []
        self.total_accuracies = [0]
        self.validation_accuracies = [0]
        self.best_val = 0
        self.best_model = copy.deepcopy(self.model)

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

    def save_plots(self):
        plt.ioff()

        ax = self.plot_loss()
        plt.savefig("Losses.png")
        plt.close()

        ax = self.plot_accuracy()
        plt.savefig("Accuracies.png")
        plt.close()

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return self.state_dict()

    def get_loss(self, data_loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        loss = 0
        with torch.inference_mode():
            for batch in data_loader:
                inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
                outputs = self(inputs)
                loss += loss_fn(outputs, y_true).item()
        return loss / len(data_loader)

    def get_accuracy(self, data_loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        correct_pred = 0
        with torch.inference_mode():
            for batch in data_loader:
                inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
                outputs = self(inputs)
                y_pred = torch.argmax(outputs, dim=1)
                correct_pred += torch.sum(y_pred == y_true).item()
        return correct_pred / len(data_loader.dataset)

    def validate(self, data_loader):
        val_loss = self.get_loss(data_loader)
        val_accuracy = self.get_accuracy(data_loader)
        return {"accuracy": val_accuracy, "loss": val_loss}

    def train_step(self, data_loader, update_every=4, verbose=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        update_every = len(data_loader) // UPDATES

        self.train()
        correct_pred = 0
        if verbose:
            print(f"Epoch: {self.epochs_trained}")
        for i, batch in enumerate(data_loader):
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
            acc_sickness = accuracy_sickness(
                y_pred, y_true
            )  # Unused now, but useful for future
            if verbose:
                print(
                    f"Batch #{i}: Loss: {loss: .5f} Accuracy: {accuracy: .4f}",
                    end="\r",
                )

                if i % update_every == 0:
                    print(f"Batch #{i}: Loss: {loss: .5f} Accuracy: {accuracy: .4f}")

        self.total_accuracies.append(correct_pred / len(data_loader.dataset))
        self.marks.append(len(self.losses))

    def train_loop(self, train_loader, val_loader, epochs, verbose=True):
        for _ in range(epochs):
            self.train_step(train_loader, verbose=verbose)
            validation = self.validate(val_loader)
            self.validation_accuracies.append(validation["accuracy"])
            self.epochs_trained += 1
            self.save_plots()

            if validation["accuracy"] > self.best_val:
                self.best_val = validation["accuracy"]
                self.best_model = copy.deepcopy(self.model)

            if validation["accuracy"] < self.total_accuracies[-1] + 0.03:
                self.model = copy.deepcopy(self.best_model)

            if verbose:
                print("VALIDATION: ", self.validate(val_loader))
        if verbose:
            print("Training Finished")


def get_smooth_losses(losses: list):
    smooth_losses = []
    for i in range(1, len(losses) + 1):
        smooth_losses.append(np.mean(losses[max(0, i - 50) : i]))
    return smooth_losses
