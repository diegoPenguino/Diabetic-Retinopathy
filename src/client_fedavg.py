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
        self.val_accuracies = []

    def plot_loss(self, ax):
        marks, val_losses = zip(*self.val_losses)
        ax.plot(marks, val_losses, "--", color="green")

    def plot_accuracy(self, ax):
        marks, val_accuracies = zip(*self.val_accuracies)
        ax.plot(marks, val_accuracies, "--", color="green")

    def append_val_metrics(self, val):
        self.val_losses.append((self.epochs_trained, val["loss"]))
        self.val_accuracies.append((self.epochs_trained, val["accuracy"]))

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
                c = 0
                for _, xd in self.get_weights().items():
                    if xd.requires_grad:
                        print(xd.grad)
                    else:
                        c += 1
                print(f"{c=}")
                self.optimizer.step()
            self.epochs_trained += 1
            val = self.validate()
            self.append_val_metrics(val)
            print(f"Epoch {epoch}: Loss: {val['loss']}, Accuracy: {val['accuracy']}")
