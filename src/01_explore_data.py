from utils.config import RAW_DATA_PATH
from utils.logger import get_logger
import pandas as pd 

logger = get_logger(__name__)

def main():
  logger.info("Loading dataset from %s", RAW_DATA_PATH)

  df = pd.read_csv(RAW_DATA_PATH)

  logger.info(f"Dataset shape (rows, columns): {df.shape}")

  logger.info("First rows of the dataset:")
  logger.info(f"\n{df.head()}")

  logger.info("Missing values per column:")
  logger.info(f"\n{df.isnull().sum()}")

  logger.info("Count of each value in the sentiments column:")
  logger.info(
      f"\n{df['is_there_an_emotion_directed_at_a_brand_or_product'].value_counts()}"
  )

  logger.info("Exploration completed.")

if __name__ == "__main__":
    main()