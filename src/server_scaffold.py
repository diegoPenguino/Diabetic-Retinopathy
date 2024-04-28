import copy
import numpy as np
from src.constants import LEARNING_RATE
from src.model import Model_Retinopathy
from src.server import Server
from src.client_scaffold import Client_Scaffold
from src.utils import get_loaders_fed, get_dataloader, split_for_federated

from src.constants import K_CLIENTS, C

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dash = "-"


class Server_Scaffold(Server):
    def __init__(self, n_clients, optimizer_fn, data_df, val_loader, lr=LEARNING_RATE):
        super(Server_Scaffold, self).__init__(
            n_clients, optimizer_fn, data_df, val_loader, lr
        )
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        for key, layer in self.model.named_parameters():
            self.control[key] = torch.zeros_like(layer.data).to(device)
            self.delta_control[key] = torch.zeros_like(layer.data).to(device)
            self.delta_y[key] = torch.zeros_like(layer.data).to(device)

        train_df = split_for_federated(data_df, n_clients)

        self.clients = [
            Client_Scaffold(self, optimizer_fn, data, val_loader, lr).to(device)
            for data in train_df
        ]
        self.clients_ind = [
            Model_Retinopathy(optimizer_fn, data, val_loader, lr).to(device)
            for data in train_df
        ]

        # self.dispatch(self.clients_id)  # THIS SHOULD DISPATCH TO INDEPENDENT ONLY!!!!!
        for client in self.clients_ind:
            client.set_weights(self.get_weights())

    def dispatch(self, clients_id):  # maybe change for .parameters() update
        for i in clients_id:
            self.clients[i].set_weights(self.get_weights())

    def client_update(self, clients_id, epochs):
        for i in clients_id:
            torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated() / (1024**2), "MB")
            print(f"Client {i}:")
            self.clients[i].train_loop(epochs)

    def train_independent(self, clients_id, epochs):
        print("Training independent clients")
        for i in clients_id:
            torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated() / (1024**2), "MB")
            print(f"Client {i}:")
            self.clients_ind[i].train_loop(epochs)

    def aggregation(self, clients_id):
        s = sum([self.clients[i].get_data_len() for i in clients_id])
        x = {}
        c = {}

        for key, layer in self.clients[0].model.named_parameters():
            x[key] = torch.zeros_like(layer.data).to(device)
            c[key] = torch.zeros_like(layer.data).to(device)

        for id in clients_id:
            for key, layer in self.clients[id].model.named_parameters():
                x[key] += self.clients[id].delta_y[key] / len(clients_id)
                c[key] += self.clients[id].delta_control[key] / len(clients_id)

        # update x and c
        for key, layer in self.model.named_parameters():
            layer.data += 1 * x[key].data  # lr=1
            self.control[key].data += c[key].data * (len(clients_id) / self.n_clients)

    def train_loop(self, rounds, epochs):
        m_clients = int(max(1, K_CLIENTS * C))
        acc, loss = self.validate().values()
        self.val_losses.append(loss)
        print(f"GLOBAL Loss = {loss}, Acc = {acc}")
        for r in range(rounds):
            print(torch.cuda.memory_allocated() / (1024**2), "MB")
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
            print(torch.cuda.memory_allocated() / (1024**2), "MB")
