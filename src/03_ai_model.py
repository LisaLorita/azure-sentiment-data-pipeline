import os
import sys
import pandas as pd
from transformers import pipeline
from google import genai
from utils.config import CLEAN_DATA_PATH
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

def should_use_llm(text, score): 
    if score < 0.75: 
        return True
    
    conflictive_keywords = ["not sure", "wait", "can't wait", "although", "despite"]
    text_lower = text.lower()
    if any(word in text_lower for word in conflictive_keywords):
        return True

    return False

def get_gemini_sentiment(client, text): 
    prompt = (
        "Analyse the sentiment of this tweet."
        "Be smart with deep semantics; for example, expressions of enthusiasm such as "
        "'cannot wait' are POSITIVE, unless they refer to something negative "
        "such as 'cannot wait for bankruptcy'. "
        "Respond ONLY with one of these three English words: POSITIVE, NEGATIVE or NEUTRAL.\n"
        f"Tweet: '{text}'"
        )

    try: 
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip().upper().split()[0]
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return "ERROR_GEMINI"

def main():
    logger.info("Starting AI model initialization...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("ERROR: GEMINI_API_KEY not found in environment variables.")
        logger.info("Please run in your terminal: export GEMINI_API_KEY='your_key_here'")
        sys.exit(1)

    logger.info("API Key detected correctly.")
    logger.info("Connecting to the Gemini client...")
    gemini_client = genai.Client()
    logger.info("Gemini client initialized successfully.")

    logger.info("Loading cleaned dataset from %s", CLEAN_DATA_PATH)
    df = pd.read_csv(CLEAN_DATA_PATH)

    test_texts = df['text'].head(5).tolist()
    logger.info(f"Testing with {len(test_texts)} tweets.")
    
    logger.info("Charging local Machine Learning model (DistilBERT)...")
    local_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    logger.info("Local model loaded successfully.")

    logger.info("Analyzing sentiments with local model...")
    local_results = local_analyzer(test_texts)

    print("\n" + "="*50)
    print("PIPELINE TEST RESULTS")
    print("="*50)

    for i, (text, local_result) in enumerate(zip(test_texts, local_results)):
        local_label = local_result['label']
        local_score = local_result['score']

        route_to_gemini = should_use_llm(text, local_score)

        print(f"\nTweet {i+1}: {text}")

        if route_to_gemini:
            print(f" -> Local (DistilBERT): {local_label} ({local_score:.2%}) [LOW TRUST OR CONFLICTIVE]")
            print(" -> [ROUTING TO GEMINI because of low trust]...")
            
            # Gemini API call to get the sentiment
            llm_result = get_gemini_sentiment(gemini_client, text)
            print(f" -> Final Result (Gemini): {llm_result} ✅")
        else:
            # Printing local result with a checkmark if we are confident about it
            print(f" -> Local (DistilBERT): {local_label} ({local_score:.2%}) [LOCAL PROCESSED] ✅")
            
        print("-" * 80)
        
    logger.info("Inference process completed successfully.")

if __name__ == "__main__":
    main()
