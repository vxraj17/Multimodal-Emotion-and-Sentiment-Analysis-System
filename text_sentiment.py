# from nltk.sentiment import SentimentIntensityAnalyzer

# def analyze_sentiment(text):
#     sia = SentimentIntensityAnalyzer()
#     sentiment_scores = sia.polarity_scores(text)

#     compound = sentiment_scores['compound']
#     if compound >= 0.05:
#         sentiment = 'Positive'
#     elif compound <= -0.05:
#         sentiment = 'Negative'
#     else:
#         sentiment = 'Neutral'

#     return sentiment, sentiment_scores
# without emoji ☝



from nltk.sentiment import SentimentIntensityAnalyzer
import emoji

def preprocess_emoji(text):
    # Convert emojis to text description (e.g., 😊 → :smiling_face_with_smiling_eyes:)
    return emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()

    # Preprocess emojis to text labels
    processed_text = preprocess_emoji(text)
    sentiment_scores = sia.polarity_scores(processed_text)

    compound = sentiment_scores['compound']
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, sentiment_scores
