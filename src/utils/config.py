from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_PATH = DATA_DIR / "train.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_train.csv"