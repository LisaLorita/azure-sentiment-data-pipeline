import pandas as pd
from utils.config import CLEAN_DATA_PATH
from transformers import pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

def main(): 
    logger.info("Charging cleaned dataset from %s", CLEAN_DATA_PATH)
    df  = pd.read_csv(CLEAN_DATA_PATH)

    logger.info("Inizializing sentiment analysis (HuggingFace)...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    logger.info("Selecting 5 sample texts for testing the AI model...")
    sample_texts = df['text'].head(5).tolist()

    logger.info("Predicting sentiment for the sample texts...")
    results = sentiment_analyzer(sample_texts)

    print("\n" + "="*30 + " AI Model Predictions " + "="*30 + "\n")

    logger.info("Predictions completed. Displaying results:")

    for text, result in zip(sample_texts, results):
        logger.info(f"Tweet: {text}")
        logger.info(f"AI model prediction: {result['label']} (confidence: {result['score']:.2%})")
        logger.info("-" * 30)

    logger.info("Process completed.")

if __name__ == "__main__":
    main()
