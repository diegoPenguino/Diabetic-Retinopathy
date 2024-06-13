from matplotlib import pyplot as plt
from src.model import Model_Retinopathy
from src.constants import LEARNING_RATE

import os

from src.constants import plot_title, plot_title


class Server(Model_Retinopathy):
    def __init__(self, n_clients, optimizer_fn, train_df, val_loader, lr=LEARNING_RATE):
        super(Server, self).__init__(optimizer_fn, train_df, val_loader, lr)
        self.marks = []
        self.is_server = True
        self.n_clients = n_clients
        self.clients_id = list(range(n_clients))
        self.title_plot = ""
        self.train_loader = None
        self.algorithm = "None"

    def plot_loss(self):  ## Change for old one
        fig, ax = plt.subplots()
        add_label = True
        for ind_client in self.clients_ind:
            ind_client.plot_val_loss(ax, add_label)
            add_label = False
        add_label = True
        for client in self.clients:
            client.plot_loss(ax, add_label)
            add_label = False
        ax.plot(
            self.marks,
            self.val_losses,
            "o-",
            label="Global Model",
            color="Blue",
            linewidth=5,
        )
        ax.set_title("Loss over epochs, " + plot_title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        return ax

    def plot_accuracy(self):
        fig, ax = plt.subplots()
        add_label = True
        for ind_client in self.clients_ind:
            ind_client.plot_val_accuracy(ax, add_label)
            add_label = False
        add_label = True
        for client in self.clients:
            client.plot_accuracy(ax, add_label)
            add_label = False
        ax.plot(
            self.marks,
            self.val_accuracies,
            "o-",
            label="Global Model",
            color="Blue",
            linewidth=5,
        )
        ax.set_title(f"Accuracy over epochs, " + plot_title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        return ax

    def plot_recall(self):
        fig, ax = plt.subplots()
        add_label = True
        for ind_client in self.clients_ind:
            ind_client.plot_val_recall(ax, add_label)
            add_label = False
        add_label = True
        for client in self.clients:
            client.plot_recall(ax, add_label)
            add_label = False
        ax.plot(
            self.marks,
            self.val_recall,
            "o-",
            label="Global Model",
            color="Blue",
            linewidth=5,
        )
        ax.set_title(f"Recall over epochs, " + plot_title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Recall")
        ax.legend()
        return ax

    def plot_precision(self):
        fig, ax = plt.subplots()
        add_label = True
        for ind_client in self.clients_ind:
            ind_client.plot_val_precision(ax, add_label)
            add_label = False
        add_label = True
        for client in self.clients:
            client.plot_precision(ax, add_label)
            add_label = False
        ax.plot(
            self.marks,
            self.val_precision,
            "o-",
            label="Global Model",
            color="Blue",
            linewidth=5,
        )
        ax.set_title(f"Precision over epochs, " + plot_title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Precision")
        ax.legend()
        return ax

    def plot_f1(self):
        fig, ax = plt.subplots()
        add_label = True
        for ind_client in self.clients_ind:
            ind_client.plot_val_f1(ax, add_label)
            add_label = False
        add_label = True
        for client in self.clients:
            client.plot_f1(ax, add_label)
            add_label = False
        ax.plot(
            self.marks,
            self.val_f1,
            "o-",
            label="Global Model",
            color="Blue",
            linewidth=5,
        )
        ax.set_title(f"F1 over epochs, " + plot_title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("F1")
        ax.legend()
        return ax
    

    def plot_bin_accuracy(self):
        fig, ax = plt.subplots()
        add_label = True
        for ind_client in self.clients_ind:
            ind_client.plot_val_bin_accuracy(ax, add_label)
            add_label = False
        add_label = True
        for client in self.clients:
            client.plot_bin_accuracy(ax, add_label)
            add_label = False
        ax.plot(
            self.marks,
            self.val_bin_accuracy,
            "o-",
            label="Global Model",
            color="Blue",
            linewidth=5,
        )
        ax.set_title(f"Binary Accuracy over epochs, " + plot_title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Binary Accuracy")
        ax.legend()
        return ax

    def save_plots(self):
        plt.ioff()
        try:
            os.mkdir(plot_title)
        except OSError as error:
            pass

        ax = self.plot_loss()
        plt.savefig(f"{plot_title}/{self.algorithm}_Losses.png")
        plt.close()

        ax = self.plot_accuracy()
        plt.savefig(f"{plot_title}/{self.algorithm}_Accuracy.png")
        plt.close()

        ax = self.plot_precision()
        plt.savefig(f"{plot_title}/{self.algorithm}_Precision.png")
        plt.close()

        ax = self.plot_recall()
        plt.savefig(f"{plot_title}/{self.algorithm}_Recall.png")
        plt.close()

        ax = self.plot_f1()
        plt.savefig(f"{plot_title}/{self.algorithm}_F1.png")
        plt.close()

        ax = self.plot_bin_accuracy()
        plt.savefig(f"{plot_title}/{self.algorithm}_Bin_Accuracy.png")
        plt.close()

    def print_results(self, rounds_taken, just_converge=False):
        try:
            os.mkdir(plot_title)
        except OSError as error:
            pass
        file_path = f"{plot_title}/{self.algorithm}_results.txt"
        with open(file_path, "w") as f:
            val = self.validate()
            print(f"Results after {rounds_taken} rounds:", file=f)
            print("Global Model results:", file=f)
            print(val, file=f)
            print("\n", file=f)
            if just_converge:
                return # No need to print individual clients results
            print("Independent clients results:", file=f)
            for i, client in enumerate(self.clients_ind):
                val = client.validate()
                print(f"Client {i}: {val}", file=f)
