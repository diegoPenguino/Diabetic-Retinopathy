from src.constants import LEARNING_RATE
from src.model import Model_Retinopathy

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()


class Client_FedAVG(Model_Retinopathy):
    def __init__(self, server, optimizer_fn, data_df, val_loader, lr=LEARNING_RATE):
        super(Client_FedAVG, self).__init__(optimizer_fn, data_df, val_loader, lr)
        self.server = server
        self.is_server = False

    def plot_loss(self, ax, add_label=False):
        marks, val_losses = zip(*self.val_losses)
        if add_label:
            ax.plot(marks, val_losses, "--", color="green", label="Clients", alpha=0.6)
            return
        ax.plot(marks, val_losses, "--", color="green", alpha=0.6)

    def plot_accuracy(self, ax, add_label=False):
        marks, val_accuracies = zip(*self.val_accuracies)
        if add_label:
            ax.plot(
                marks, val_accuracies, "--", color="green", label="Clients", alpha=0.6
            )
            return
        ax.plot(marks, val_accuracies, "--", color="green", alpha=0.6)

    def plot_recall(self, ax, add_label=False):
        marks, val_recall = zip(*self.val_recall)
        if add_label:
            ax.plot(marks, val_recall, "--", color="green", label="Clients", alpha=0.6)
            return
        ax.plot(marks, val_recall, "--", color="green", alpha=0.6)

    def plot_precision(self, ax, add_label=False):
        marks, val_precision = zip(*self.val_precision)
        if add_label:
            ax.plot(
                marks, val_precision, "--", color="green", label="Clients", alpha=0.6
            )
            return
        ax.plot(marks, val_precision, "--", color="green", alpha=0.6)

    def plot_f1(self, ax, add_label=False):
        marks, val_f1 = zip(*self.val_f1)
        if add_label:
            ax.plot(marks, val_f1, "--", color="green", label="Clients", alpha=0.6)
            return
        ax.plot(marks, val_f1, "--", color="green", alpha=0.6)

    def plot_bin_accuracy(self, ax, add_label=False):
        marks, val_bin_accuracy = zip(*self.val_bin_accuracy)
        if add_label:
            ax.plot(
                marks, val_bin_accuracy, "--", color="green", label="Clients", alpha=0.6
            )
            return
        ax.plot(marks, val_bin_accuracy, "--", color="green", alpha=0.6)

    def append_val_metrics(self, val):
        self.val_losses.append((self.epochs_trained, val["loss"]))
        self.val_accuracies.append((self.epochs_trained, val["accuracy"]))
        self.val_recall.append((self.epochs_trained, val["recall"]))
        self.val_precision.append((self.epochs_trained, val["precision"]))
        self.val_f1.append((self.epochs_trained, val["f1"]))
        self.val_bin_accuracy.append((self.epochs_trained, val["accuracy_binary"]))

    def train_loop(self, epochs):
        val = self.validate()
        self.append_val_metrics(val)

        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, y_true)
                loss.backward()
                self.optimizer.step()
            self.epochs_trained += 1
            val = self.validate()
            self.append_val_metrics(val)
            print(f"Epoch {epoch}: {val}")
        self.model.zero_grad()
