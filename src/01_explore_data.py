from utils.config import RAW_DATA_PATH
import pandas as pd 

df = pd.read_csv(RAW_DATA_PATH)

print("Dataset shape (rows, columns):")
print(df.shape)

print(df.head())

print(df.isnull().sum())

print(df['is_there_an_emotion_directed_at_a_brand_or_product'].value_counts())