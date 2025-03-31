# Sentiment Analysis

## Business Context: Amazon Product Reviews

Amazon analyzes millions of customer reviews to:
- Identify products with satisfaction issues
- Highlight top-performing products
- Monitor customer sentiment trends over time
- Provide personalized recommendations based on sentiment patterns

## What is Sentiment Analysis?

Sentiment analysis determines the emotional tone behind text:
- Positive: "This laptop exceeds my expectations and the battery life is amazing!"
- Negative: "The product broke after just two uses and customer service was unhelpful."
- Neutral: "The package arrived on Tuesday. It contains the items I ordered."

## Notebook Demonstration

We'll build a simple sentiment classifier using:
- A dataset of product reviews with ratings
- Text preprocessing (tokenization, removing stopwords)
- Feature extraction with TF-IDF vectorization
- A machine learning classifier (Naive Bayes)
- Evaluation metrics to assess performance

```python
# Sample code (from the notebook)
# Load and prepare data
import pandas as pd
from sklearn.model_selection import train_test_split

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Model training
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create a pipeline for sentiment classification
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])

# Train the model
sentiment_pipeline.fit(X_train, y_train)

# Make predictions
predictions = sentiment_pipeline.predict(X_test)
```

## Interpreting Results

- Our model identifies positive and negative sentiment with ~85% accuracy
- We can apply this to new, unseen reviews to gauge customer satisfaction
- The model highlights which words strongly indicate positive or negative sentiment
- Business value: Automatically flag negative reviews for customer service follow-up

## Learning Challenge

Try applying our sentiment classifier to these reviews:
1. "The product works as described but shipping took longer than expected."
2. "While the design is beautiful, the functionality is disappointing."

What do you think makes sentiment analysis challenging in these examples? How might businesses address these nuances?

## Real-world Implementation

- Companies typically use more sophisticated models (BERT, RoBERTa)
- Cloud services provide pre-trained sentiment analysis APIs
- Custom training on industry-specific language improves accuracy
- Multi-class sentiment (very negative to very positive) offers more granularity 