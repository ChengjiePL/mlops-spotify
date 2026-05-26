import glob
import pandas as pd

def load_data(path):
    files = glob.glob(path)

    if len(files) == 0:
        raise FileNotFoundError(f"No files found for pattern: {path}")

    return pd.read_csv(files[0])
