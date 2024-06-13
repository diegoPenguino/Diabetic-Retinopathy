import copy
import numpy as np
from src.constants import LEARNING_RATE
from src.model import Model_Retinopathy
from src.server import Server
from src.client_fedavg import Client_FedAVG
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


class Server_FedAVG(Server):
    def __init__(self, n_clients, optimizer_fn, data_df, val_loader, lr=LEARNING_RATE):
        super(Server_FedAVG, self).__init__(
            n_clients, optimizer_fn, data_df, val_loader, lr
        )
        self.algorithm = "FedAvg"
        ## Random split
        if iid:
            train_df = split_for_federated(data_df, n_clients)
        else: # FOR NON IID:
            train_df = split_non_iid(data_df)
            self.n_clients = len(train_df)
            self.clients_id = list(range(self.n_clients))

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
        m_clients = int(max(1, self.n_clients * C))
        if len(self.val_losses) == 0:
            val = self.validate()
            self.append_val_metrics(val)
            print(f"GLOBAL: {val}")
        rounds_taken = None
        for r in range(rounds):
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
            elif acc > 0.7 and recall > 0.9 and precision > 0.9 and f1 > 0.9:
                print(f"DONE in {rounds_taken} rounds")
                break
            gpu_memory = torch.cuda.memory_allocated()
            print(gpu_memory / (1024**2), "MB used")
        self.print_results(rounds_taken, just_converge)
