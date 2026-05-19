from utils.config import CLEAN_DATA_PATH
import pandas as pd
from transformers import pipeline

df  = pd.read_csv(CLEAN_DATA_PATH)

sentiment_analyzer = pipeline("sentiment-analysis")

textos_prueba = df['text'].head(5).tolist()

resultados = sentiment_analyzer(textos_prueba)

for texto, resultado in zip(textos_prueba, resultados):
    print(f"Tweet: {texto}")
    print(f"Predicción IA: {resultado}")
    print("-" * 30)
