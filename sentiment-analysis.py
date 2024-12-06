
# 1. Import Libraries
import pandas as pd
from textblob import TextBlob  
import matplotlib.pyplot as plt

# 2. Load the Preprocessed Data
data_path = "path_to_your_preprocessed_data.csv"
news_data = pd.read_csv(data_path)

# 3. Sentiment Analysis Function
def get_sentiment(text):
    """Calculate the sentiment polarity of a given text using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity

# 4. Apply Sentiment Analysis
news_data['sentiment_score'] = news_data['processed_text'].apply(lambda x: get_sentiment(" ".join(eval(x))))

# 5. Categorize Sentiment
def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

news_data['sentiment_category'] = news_data['sentiment_score'].apply(categorize_sentiment)

# 6. Save the Results
news_data.to_csv("news_with_sentiment.csv", index=False)

# 7. Visualization
category_counts = news_data['sentiment_category'].value_counts()
category_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Categories in News Articles')
plt.show()
