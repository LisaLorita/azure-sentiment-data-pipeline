import pandas as pd
from utils.config import CLEAN_DATA_PATH
from transformers import pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

def main(): 
    logger.info("Charging cleaned dataset from %s", CLEAN_DATA_PATH)
    df  = pd.read_csv(CLEAN_DATA_PATH)

    logger.info("Inizializing sentiment analysis (HuggingFace)...")
    sentiment_analyzer = pipeline("sentiment-analysis")

    logger.info("Selecting 5 sample texts for testing the AI model...")
    sample_texts = df['text'].head(5).tolist()

    logger.info("Predicting sentiment for the sample texts...")
    results = sentiment_analyzer(sample_texts)

    logger.info("Predictions completed. Displaying results:")

    for text, resultado in zip(sample_texts, results):
        logger.info(f"Tweet: {text}")
        logger.info(f"AI model prediction: {resultado}")
        logger.info("-" * 30)

    logger.info("Process completed.")

if __name__ == "__main__":
    main()
