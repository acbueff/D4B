# Sentiment Analysis for Product Reviews

---

## Introduction to Sentiment Analysis

- **What is Sentiment Analysis?** 
  - Computational technique to identify and extract subjective information
  - Determines whether text expresses positive, negative, or neutral sentiment
  - Can detect emotional tone, opinion, or attitude

- **Business Application: Amazon Product Reviews**
  - Amazon processes millions of product reviews daily
  - Sentiment analysis helps categorize customer feedback automatically
  - Enables data-driven product improvements and recommendation systems

---

## How Sentiment Analysis Works

- **Text Pre-processing Steps:**
  - Tokenization - Breaking text into words/tokens
  - Removing stopwords and punctuation
  - Lemmatization/stemming - Reducing words to base forms
  - Converting text to numerical features (vectorization)

- **Classification Approaches:**
  - Dictionary/Lexicon-based methods
  - Machine Learning classifiers (Naive Bayes, SVM, etc.)
  - Deep Learning techniques (LSTM, BERT, etc.)

---

## Practical Implementation

```python
# Example code snippet:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create a pipeline with vectorizer and classifier
sentiment_model = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# Train the model
sentiment_model.fit(training_reviews, training_sentiments)

# Predict sentiment on new reviews
predictions = sentiment_model.predict(test_reviews)
```

---

## Feature Engineering for Sentiment

- **Bag of Words (BoW) vs TF-IDF**
  - BoW: Simple word count
  - TF-IDF: Term frequency × inverse document frequency
  
- **N-grams capture phrases**
  - Unigrams: "not", "good" (loses context)
  - Bigrams: "not good" (preserves context)

- **Additional Features**
  - Exclamation points count
  - Capitalized words
  - Emoticons and emojis

---

## Evaluating Sentiment Models

- **Common Metrics:**
  - Accuracy: Overall correct predictions
  - Precision: Correctness of positive predictions
  - Recall: Ability to find all positive reviews
  - F1-Score: Harmonic mean of precision and recall

- **Business Considerations:**
  - False positives vs. false negatives
  - Adjusting classification thresholds based on business needs

---

## Amazon's Implementation in Practice

- **Scale Considerations:**
  - Processing millions of reviews efficiently
  - Multi-language support
  - Real-time vs. batch processing

- **Business Integration:**
  - Feedback loops to product teams
  - Customer service prioritization
  - Review highlights on product pages
  - Input for recommendation algorithms

---

## Beyond Binary Sentiment

- **Fine-grained Sentiment Analysis:**
  - 5-star rating prediction
  - Aspect-based sentiment analysis
    - "Battery life is excellent but camera quality is poor"
  
- **Emotion Detection:**
  - Identifying specific emotions (anger, joy, disappointment)
  - Understanding customer emotional journey

---

## Learning Challenge

**Exercise:** Analyze these product reviews and predict their sentiment:

1. "This laptop exceeded my expectations, highly recommend!"
2. "Received the product damaged. Customer service was unhelpful."
3. "Average performance for the price. Nothing special but does the job."

**Discussion Questions:**
- How might ambiguous reviews affect Amazon's algorithms?
- What features would help classify the third review correctly?
- How could sentiment analysis improve product development?

---

## Key Takeaways

- Sentiment analysis automatically extracts opinions from text
- Amazon uses this technology to process millions of reviews
- Practical implementation requires careful preprocessing and feature engineering
- Business value comes from integrating insights across the organization
- Modern approaches increasingly focus on context and specific aspects of products 