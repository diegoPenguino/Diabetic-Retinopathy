import copy
import numpy as np
from src.constants import LEARNING_RATE
from src.model import Model_Retinopathy
from src.server import Server
from src.client_scaffold import Client_Scaffold
from src.utils import (
    get_loaders_fed,
    get_dataloader,
    split_for_federated,
    split_non_iid,
)

from src.constants import K_CLIENTS, C, just_converge, iid

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dash = "-"


class Server_Scaffold(Server):
    def __init__(self, n_clients, optimizer_fn, data_df, val_loader, lr=LEARNING_RATE):
        super(Server_Scaffold, self).__init__(
            n_clients, optimizer_fn, data_df, val_loader, lr
        )
        self.algorithm = "SCAFFOLD"
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

        weights = self.get_weights()
        for key, layer in self.model.named_parameters():
            self.control[key] = torch.zeros_like(layer.data).float().to("cpu")
            self.delta_control[key] = torch.zeros_like(layer.data).float().to("cpu")
        for key, layer in weights.items():
            self.delta_y[key] = torch.zeros_like(layer.data).float().to("cpu")

        if iid:
            train_df = split_for_federated(data_df, n_clients)
        else:  # FOR NON IID:
            train_df = split_non_iid(data_df)
            self.n_clients = len(train_df)
            self.clients_id = list(range(self.n_clients))

        self.clients = [
            Client_Scaffold(self, optimizer_fn, data, val_loader, lr)
            for data in train_df
        ]
        self.clients_ind = [
            Model_Retinopathy(optimizer_fn, data, val_loader, lr) for data in train_df
        ]

        # self.dispatch(self.clients_id)  # THIS SHOULD DISPATCH TO INDEPENDENT ONLY!!!!!
        for client in self.clients_ind:
            client.set_weights(self.get_weights())

        self.title_plot = "SCAFFOLD"

    def dispatch(self, clients_id):
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
        Dx = {}
        Dc = {}

        for key, layer in self.clients[0].model.named_parameters():
            Dc[key] = torch.zeros_like(layer.data).float().to("cpu")
        for key, layer in self.get_weights().items():
            Dx[key] = torch.zeros_like(layer.data).float().to("cpu")

        for id in clients_id:
            actual_client = self.clients[id]
            for key, layer in actual_client.model.named_parameters():
                Dc[key] += (
                    actual_client.delta_control[key] * actual_client.get_data_len() / s
                )
            for key, layer in self.get_weights().items():
                Dx[key] += actual_client.delta_y[key] * actual_client.get_data_len() / s

        # update x and c
        weights = {}
        for key, layer in self.get_weights().items():
            weights[key] = layer.data.clone().float().to("cpu")

        for key, layer in self.model.named_parameters():
            self.control[key].data += Dc[key].data * (len(clients_id) / self.n_clients)

        for key, layer in self.get_weights().items():
            weights[key] += 1 * Dx[key].data  # lr=1
            weights[key] = weights[key].to(device)
        self.set_weights(weights)

    def train_loop(self, rounds, epochs):
        m_clients = int(max(1, self.n_clients * C))
        if len(self.val_losses) == 0:
            val = self.validate()
            self.append_val_metrics(val)
            print(f"GLOBAL: {val}")
        rounds_taken = None
        for r in range(rounds):
            print(torch.cuda.memory_allocated() / (1024**2), "MB")
            print(f"Round {r}\n{dash*50}")
            selected_clients = np.random.choice(
                self.clients_id, m_clients, replace=False
            )
            selected_clients.sort()
            self.dispatch(selected_clients)
            self.client_update(selected_clients, epochs)
            if not just_converge:
                self.train_independent(selected_clients, epochs)
            self.aggregation(selected_clients)
            val = self.validate()
            self.epochs_trained += epochs
            self.append_val_metrics(val)
            print(f"GLOBAL: {val}")
            rounds_taken = r + 1
            acc = val["accuracy"]
            recall = val["recall"]
            precision = val["precision"]
            f1 = val["f1"]
            if not just_converge:
                self.save_plots()
            elif acc > 0.7 and recall > 0.90 and precision > 0.9 and f1 > 0.9:
                print(f"DONE in {rounds_taken} rounds")
                break
            print(torch.cuda.memory_allocated() / (1024**2), "MB")
        self.print_results(rounds_taken, just_converge)
