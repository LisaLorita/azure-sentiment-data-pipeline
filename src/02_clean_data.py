import pandas as pd 
import re 
from utils.config import RAW_DATA_PATH, CLEAN_DATA_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

def clean_text(text):
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def main():
    logger.info("Inizializing data cleaning process...")    
    logger.info("Loading cleaning dataset from %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Renaming columns for consistency...")
    df = df.rename(columns={
        'tweet_text': 'text', 
        'emotion_in_tweet_is_directed_at': 'brand',
        'is_there_an_emotion_directed_at_a_brand_or_product': 'sentiment'
    })

    logger.info("Removing rows with missing text...")
    df = df.dropna(subset=['text'])

    logger.info("Removing rows with ambiguous sentiment...")
    df = df[df['sentiment'] != "I can't tell"]

    logger.info("Applying regex for cleaning text data...")
    df['text'] = df['text'].apply(clean_text)

    logger.info("Saving cleaned dataset to %s", CLEAN_DATA_PATH)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    logger.info(f"Process completed. Dataset shape: {df.shape[0]}")

if __name__ == "__main__":
    main()