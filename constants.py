## CONSTANTS
INPUT_SHAPE = (224, 224, 3)

YEAR = 2019
ext = "png" if YEAR == 2019 else "jpeg"

directory = f"data/Resized/preprocessed_{YEAR}_ben_color"
csv_file = f"data/Resized/train_{YEAR}.csv"

## MODEL CONSTANTS
EPOCHS = 8
UPDATES = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

K_CLIENTS = 10
C = 1  # C: Fraction of clients to be selected
rounds = 30  # T: Number of rounds
clients = list(range(K_CLIENTS))
