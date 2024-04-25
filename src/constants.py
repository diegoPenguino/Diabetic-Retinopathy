## CONSTANTS
INPUT_SHAPE = (224, 224, 3)

YEAR = 2019
ext = "png" if YEAR == 2019 else "jpeg"

directory = f"data/Resized/preprocessed_{YEAR}_ben_color"
csv_file = f"data/Resized/train_{YEAR}.csv"

## MODEL CONSTANTS
EPOCHS = 10
UPDATES = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
