import copy
from src.constants import LEARNING_RATE
from src.model import Model_Retinopathy
from src.scaffold_optimizer import Scaffold_Optimizer


import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()


class Client_Scaffold(Model_Retinopathy):
    def __init__(self, server, optimizer_fn, data_df, val_loader, lr=LEARNING_RATE):
        super(Client_Scaffold, self).__init__(optimizer_fn, data_df, val_loader, lr)
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        weights = self.get_weights()
        for key, layer in self.model.named_parameters():
            self.control[key] = torch.zeros_like(layer.data).float().to("cpu")
            self.delta_control[key] = torch.zeros_like(layer.data).float().to("cpu")
        for key, layer in weights.items():
            self.delta_y[key] = torch.zeros_like(layer.data).float().to("cpu")
        self.server = server
        self.is_server = False
        self.val_accuracies = []
        self.optimizer = Scaffold_Optimizer(self.model.parameters(), lr=lr)

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
        x_model = copy.deepcopy(self.model).to("cpu")
        # x_weights = copy.deepcopy(self.model).to("cpu")
        x_control = copy.deepcopy(self.control)

        self.optimizer = Scaffold_Optimizer(self.model.parameters(), lr=self.lr)

        for epoch in range(epochs):
            print(torch.cuda.memory_allocated() / (1024**2), "MB")
            self.model.train()
            for batch in self.train_loader:
                inputs, y_true = batch["image"].to(device), batch["labels"].to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, y_true)
                loss.backward()
                """ for param, c, ci in zip(
                    self.model.parameters(),
                    self.server.control.values(),
                    self.control.values(),
                ):
                    param.grad.data = (
                        param.grad.data + c.data.to(device) - ci.data.to(device)
                    ) """
                self.optimizer.step(self.server.control, self.control)
                del inputs
                del y_true
                torch.cuda.empty_cache()
            self.epochs_trained += 1
            val = self.validate()
            self.append_val_metrics(val)
            print(f"Epoch {epoch}: Loss: {val['loss']}, Accuracy: {val['accuracy']}")

        temp = {}
        weights = self.get_weights()
        for key, layer in weights.items():
            temp[key] = layer.data.clone().float().to("cpu")

        local_steps = epochs * len(self.train_loader)
        for key, layer in x_model.state_dict().items():
            self.delta_y[key] = temp[key] - layer.data
            if key in dict(x_model.named_parameters()).keys():
                self.control[key] = (
                    self.control[key]
                    - self.server.control[key]
                    + (layer.data - temp[key]) / (local_steps * self.lr)
                )
                self.delta_control[key] = self.control[key] - x_control[key]
