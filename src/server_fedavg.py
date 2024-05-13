import copy
import numpy as np
from src.constants import LEARNING_RATE
from src.model import Model_Retinopathy
from src.server import Server
from src.client_fedavg import Client_FedAVG
from src.utils import get_loaders_fed, get_dataloader, split_for_federated

from src.constants import K_CLIENTS, C

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dash = "-"


class Server_FedAVG(Server):
    def __init__(self, n_clients, optimizer_fn, data_df, val_loader, lr=LEARNING_RATE):
        super(Server_FedAVG, self).__init__(
            n_clients, optimizer_fn, data_df, val_loader, lr
        )

        train_df = split_for_federated(data_df, n_clients)

        self.clients = [
            Client_FedAVG(self, optimizer_fn, data, val_loader, lr) for data in train_df
        ]
        self.clients_ind = [
            Model_Retinopathy(optimizer_fn, data, val_loader, lr) for data in train_df
        ]

        # self.dispatch(self.clients_id)  # THIS SHOULD DISPATCH TO INDEPENDENT ONLY!!!!!
        for client in self.clients_ind:
            client.set_weights(self.get_weights())

        self.title_plot = "FedAVG"

    def dispatch(self, clients_id):  # maybe change for .parameters() update
        for i in clients_id:
            self.clients[i].set_weights(self.get_weights())

    def client_update(self, clients_id, epochs):
        for i in clients_id:
            print(f"Client {i}:")
            self.clients[i].train_loop(epochs)

    def train_independent(self, clients_id, epochs):
        print("Training independent clients")
        for i in clients_id:
            print(f"Client {i}:")
            self.clients_ind[i].train_loop(epochs)

    def aggregation(self, clients_id):
        s = sum([self.clients[i].get_data_len() for i in clients_id])
        print(f"s=")
        new_weights = {}
        weights = self.get_weights()
        for key, layer in weights.items():
            new_weights[key] = torch.zeros_like(layer.data).float().to(device)
        for i in clients_id:
            for key, layer in self.clients[i].get_weights().items():
                new_weights[key] += (
                    layer.data.float() * self.clients[i].get_data_len() / s
                )
        self.set_weights(new_weights)

    def train_loop(self, rounds, epochs):
        m_clients = int(max(1, K_CLIENTS * C))
        if len(self.val_losses) == 0:
            acc, loss = self.validate().values()
            self.val_losses.append(loss)
            self.val_accuracies.append(acc)
            print(f"GLOBAL Loss = {loss}, Acc = {acc}")
        for r in range(rounds):
            print(f"Round {r}\n{dash*50}")
            selected_clients = np.random.choice(
                self.clients_id, m_clients, replace=False
            )
            selected_clients.sort()

            self.dispatch(selected_clients)
            self.client_update(selected_clients, epochs)
            self.train_independent(selected_clients, epochs)
            self.aggregation(selected_clients)
            acc, loss = self.validate().values()
            self.val_losses.append(loss)
            self.val_accuracies.append(acc)
            self.marks.append(self.marks[-1] + epochs)
            print(f"GLOBAL Loss = {loss}, Acc = {acc}")
            self.save_plots()
            gpu_memory = torch.cuda.memory_allocated()
            print(gpu_memory / (1024**2), "MB used")
