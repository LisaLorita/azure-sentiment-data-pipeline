import pandas as pd 
import re 

def clean_text(text):
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

df = pd.read_csv('data/train.csv')

df = df.rename(columns={
    'tweet_text': 'text', 
    'emotion_in_tweet_is_directed_at': 'brand',
    'is_there_an_emotion_directed_at_a_brand_or_product': 'sentiment'
})

df = df.dropna(subset=['text'])

df = df[df['sentiment'] != "I can't tell"]

df['text'] = df['text'].apply(clean_text)

df.to_csv('data/clean_train.csv', index=False)


print(df.shape)
print(df.head())

