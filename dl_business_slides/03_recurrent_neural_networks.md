# Recurrent Neural Networks (RNNs)

---

## Introduction to Sequential Data

- **Sequential Data in Business:**
  - Time series: Stock prices, sales figures, sensor readings
  - Text: Customer reviews, support tickets, contracts
  - User behavior: Click streams, purchase history, app usage

- **Challenges of Sequential Data:**
  - Variable length sequences
  - Long-term dependencies
  - Order matters
  - Context influences interpretation

- **Why Traditional NNs Struggle:**
  - Fixed input size
  - No memory of previous inputs
  - Treat inputs as independent

---

## Recurrent Neural Networks: The Concept

- **Key Innovation:** Maintain internal memory state

- **Architecture:**
  - Process one element of sequence at a time
  - Update hidden state with each element
  - Share same weights across all time steps
  - Feed hidden state back into network

- **Mathematical Representation:**
  - h<sub>t</sub> = f(W<sub>h</sub>h<sub>t-1</sub> + W<sub>x</sub>x<sub>t</sub> + b)
  - y<sub>t</sub> = g(W<sub>y</sub>h<sub>t</sub> + b<sub>y</sub>)

---

## RNN Architecture Diagram

![RNN Architecture](https://miro.medium.com/max/1400/1*WMnFSJHzOloFlJHU6fVN-g.gif)

- **Left:** Unfolded representation (over time)
- **Right:** Compact representation

Key elements:
- x<sub>t</sub>: Input at time step t
- h<sub>t</sub>: Hidden state at time step t
- y<sub>t</sub>: Output at time step t
- A: Recurrent weights (shared across time)

---

## Types of RNN Architectures

- **Many-to-Many:**
  - Input: Sequence → Output: Sequence
  - Example: Machine translation, text generation
  - Business use: Converting speech to text for customer service

- **Many-to-One:**
  - Input: Sequence → Output: Single value
  - Example: Sentiment analysis, sequence classification
  - Business use: Fraud detection from transaction sequences

- **One-to-Many:**
  - Input: Single value → Output: Sequence
  - Example: Image captioning, music generation
  - Business use: Generating product descriptions from images

---

## Basic RNN Implementation

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate through RNN
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        # For sequence classification, we only need the last output
        out = self.fc(out[:, -1, :])
        return out
```

---

## The Vanishing Gradient Problem in RNNs

- **Problem:** Gradients diminish exponentially with sequence length
  - Makes learning long-term dependencies difficult
  - Practical limit of ~10-20 time steps

- **Business Impact:**
  - Inability to capture long-term patterns
  - Poor performance on long sequences
  - Limited memory of past events

- **Example:**
  - Fraud detection: Missing patterns spanning multiple weeks
  - Customer journey analysis: Unable to connect early interactions with purchases

---

## Long Short-Term Memory (LSTM) Networks

- **Solution to Vanishing Gradients:** More sophisticated memory cell

- **LSTM Components:**
  - **Forget Gate:** Controls what to remove from cell state
  - **Input Gate:** Controls what new information to store
  - **Output Gate:** Controls what to output based on cell state

- **Business Advantage:**
  - Capture dependencies spanning hundreds of time steps
  - Remember relevant information over long periods
  - Discard irrelevant information

---

## LSTM Cell Structure

![LSTM Cell](https://miro.medium.com/max/1400/1*yBXV9o5q7L_CvY7quJt3WQ.png)

- **Cell State (C<sub>t</sub>):** Long-term memory
- **Hidden State (h<sub>t</sub>):** Short-term memory
- **Gates:** Regulate information flow
  - f<sub>t</sub>: Forget gate
  - i<sub>t</sub>: Input gate
  - o<sub>t</sub>: Output gate

---

## LSTM Implementation

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state for the last time step
        out = self.fc(out[:, -1, :])
        return out
```

---

## Gated Recurrent Unit (GRU)

- **Simplified Alternative to LSTM:**
  - Fewer parameters than LSTM
  - Often similar performance
  - Faster training and inference

- **Key Differences:**
  - Combines forget and input gates into a "update gate"
  - Merges cell state and hidden state
  - Uses "reset gate" to control past information influence

- **Business Consideration:**
  - Good option when computational resources are limited
  - Often preferred for smaller datasets

---

## Business Application: Demand Forecasting

- **Business Problem:**
  - Accurately predicting future product demand
  - Multiple factors influence trends (seasonality, promotions, etc.)
  - Past patterns influence future behavior

- **RNN/LSTM Approach:**
  - Input: Historical sales, promotions, seasonality features
  - Model: Stacked LSTM with multiple layers
  - Output: Predicted demand for future periods

- **Business Impact:**
  - 25-30% reduction in forecast error vs. traditional methods
  - 15-20% reduction in stockouts
  - 10-15% reduction in excess inventory

---

## Business Application: Fraud Detection

- **Problem:** Detecting fraudulent transaction sequences
  - Fraudsters create patterns of legitimate transactions before fraud
  - Individual transactions may appear normal
  - Context and sequence matter

- **LSTM Solution:**
  - Process sequence of customer transactions
  - Learn patterns that precede fraudulent activity
  - Flag suspicious sequences in real-time

- **Real-world Example:**
  - Visa's neural network fraud detection system
  - Analyzes transaction sequences in real-time
  - 93% detection rate with lower false positives
  - $25 billion in prevented fraud annually

---

## Business Application: Customer Churn Prediction

- **Problem:** Predicting customer attrition
  - Engagement patterns over time indicate churn risk
  - Multiple interactions across channels
  - Early warning enables retention efforts

- **LSTM Approach:**
  - Input: Sequence of customer interactions, purchases, support contacts
  - Features: Recency, frequency, monetary value, sentiment
  - Output: Churn probability within future time window

- **Business Impact:**
  - 20% improvement over traditional models
  - $3.5M annual savings for telecom provider
  - Prioritized retention efforts for high-value customers

---

## Text Processing with RNNs

- **Business Applications:**
  - Sentiment analysis of customer reviews
  - Categorization of support tickets
  - Automatic email routing and prioritization
  - Contract clause extraction and analysis

- **Implementation Approach:**
  - Convert words to numeric vectors (embeddings)
  - Process word sequences through LSTM/GRU
  - Final hidden state represents document
  - Classify or process based on this representation

- **Business Value:**
  - Automation of text-heavy processes
  - Consistent analysis of large document volumes
  - Real-time processing of incoming communications

---

## Bidirectional RNNs

- **Limitation of Standard RNNs:**
  - Only consider past context, not future context
  - Suboptimal for tasks where complete context matters

- **Bidirectional RNN Solution:**
  - Process sequence in both directions
  - Forward RNN: Left to right
  - Backward RNN: Right to left
  - Combine both states for prediction

- **Business Applications:**
  - Improved document understanding
  - Better sentiment analysis
  - More accurate named entity recognition in contracts

---

## Bidirectional LSTM Architecture

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        
        # Note: output size is doubled because of bidirectional
        self.fc = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state for the last time step
        out = self.fc(out[:, -1, :])
        return out
```

---

## Attention Mechanisms

- **Problem:** Even LSTMs struggle with very long sequences

- **Attention Solution:**
  - Allow model to focus on relevant parts of input sequence
  - Compute importance weights for each time step
  - Create context vector as weighted sum of hidden states

- **Business Benefits:**
  - Better performance on long sequences
  - Improved interpretability: see what the model focuses on
  - Foundation for transformer models (BERT, GPT)

---

## Modern Developments: Transformers

- **Evolution Beyond RNNs:**
  - Transformers use attention without recurrence
  - Process entire sequence in parallel
  - Capture long-range dependencies better

- **Key Advantages:**
  - Faster training (parallelizable)
  - Better performance on many tasks
  - Scale to very long sequences

- **Business Applications:**
  - Large language models (GPT, BERT)
  - Document understanding and processing
  - Advanced time series forecasting

---

## Learning Challenge: Financial Time Series

**Scenario:** A financial services company wants to predict stock price movements based on historical price data and news sentiment.

**Exercise:**
1. What sequence model architecture would you recommend?
2. What features would you include in the input sequence?
3. How would you handle different time scales (daily prices vs. real-time news)?
4. What business metrics would determine success?
5. How would you explain model predictions to investment teams?

**Discussion:**
- What are the regulatory considerations for algorithmic trading models?
- How would you update the model as market conditions evolve?
- What safeguards would you implement against extreme predictions?

---

## Key Takeaways

- RNNs process sequential data by maintaining a hidden state
- LSTMs and GRUs solve the vanishing gradient problem for long sequences
- Bidirectional RNNs incorporate both past and future context
- Attention mechanisms allow focusing on relevant parts of sequences
- Business applications include time series forecasting, text analysis, and fraud detection
- Modern transformer architectures increasingly replace traditional RNNs
- Sequence models enable businesses to extract value from temporal and ordered data 