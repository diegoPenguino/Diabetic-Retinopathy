## CONSTANTS
import torch

INPUT_SHAPE = (224, 224, 3)

YEAR = 2019
ext = "png" if YEAR == 2019 else "jpeg"

directory = f"data/Resized/preprocessed_{YEAR}_ben_color"
csv_file = f"data/Resized/train_{YEAR}.csv"

optimizer_fn = torch.optim.SGD
## EXPERIMENT CONSTANTS
just_converge = False
iid = True
model_arch = "CNN"
C = 1.0
# model_arch = "ResNet"

## MODEL CONSTANTS
EPOCHS = 5
UPDATES = 4
BATCH_SIZE = 16
LEARNING_RATE = 0.005 if model_arch == "CNN" else 0.001
K_CLIENTS = 30 if iid else 20

rounds = 1000 if just_converge else 50
clients = list(range(K_CLIENTS))
if not just_converge:
    C = 1.0

name = "Regular CNN" if model_arch == "CNN" else "ResNet CNN"
iid_name = "IID" if iid else "non-IID"

if just_converge:
    plot_title = f"{name} C = {C} - {K_CLIENTS} clients {iid_name}"
else:
    plot_title = f"{name} - {K_CLIENTS} clients {iid_name}"
