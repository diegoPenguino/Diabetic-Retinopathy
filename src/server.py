from matplotlib import pyplot as plt
from src.model import Model_Retinopathy
from src.constants import LEARNING_RATE


class Server(Model_Retinopathy):
    def __init__(self, n_clients, optimizer_fn, train_df, val_loader, lr=LEARNING_RATE):
        super(Server, self).__init__(optimizer_fn, train_df, val_loader, lr)
        self.marks = [0]
        self.val_accuracies = [0]
        self.val_losses = []
        self.is_server = True
        self.n_clients = n_clients
        self.clients_id = list(range(n_clients))

        self.train_loader = None

    def plot_loss(self):  ## Change for old one
        fig, ax = plt.subplots()
        ax.plot(
            self.marks,
            self.val_losses,
            "o-",
            label="Global Model",
            color="Blue",
            linewidth=5,
        )
        for client in self.clients:
            client.plot_loss(ax)
        for ind_client in self.clients_ind:
            ind_client.plot_val_loss(ax)
        ax.set_title("Loss over epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        return ax

    def plot_accuracy(self):
        fig, ax = plt.subplots()
        ax.plot(
            self.marks,
            self.val_accuracies,
            "o-",
            label="Global Model",
            color="Blue",
            linewidth=5,
        )
        for client in self.clients:
            client.plot_accuracy(ax)
        for ind_client in self.clients_ind:
            ind_client.plot_val_accuracy(ax)
        ax.set_title("Accuracy over epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        return ax

    def save_plots(self):
        plt.ioff()

        ax = self.plot_loss()
        plt.savefig("Losses_fed.png")
        plt.close()

        ax = self.plot_accuracy()
        plt.savefig("Accuracies_fed.png")
        plt.close()
