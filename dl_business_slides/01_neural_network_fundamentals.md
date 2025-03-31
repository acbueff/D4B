# Neural Network Fundamentals

---

## The Building Blocks: Neurons

- **Biological Inspiration:**
  - Artificial neurons inspired by human brain cells
  - Receive, process, and transmit information

- **Mathematical Representation:**
  - Inputs (x): Data features
  - Weights (w): Importance of each input
  - Bias (b): Threshold adjustment
  - Activation function: Non-linear transformation
  - Output = f(Σ(w·x) + b)

- **Business Analogy:**
  - Like a weighted decision-making process with multiple factors

---

## Activation Functions: Adding Non-linearity

- **Purpose:** Enable networks to learn complex, non-linear patterns

- **Common Activation Functions:**
  - **ReLU:** f(x) = max(0, x)
    - Simple, computationally efficient
    - Industry standard for hidden layers
  
  - **Sigmoid:** f(x) = 1/(1+e^(-x))
    - Outputs between 0-1
    - Used for binary classification
  
  - **Tanh:** f(x) = (e^x - e^(-x))/(e^x + e^(-x))
    - Outputs between -1 and 1
    - Zero-centered alternative to sigmoid

---

## Network Architecture

- **Layers in a Neural Network:**
  - **Input Layer:** Receives raw data
  - **Hidden Layers:** Internal processing
  - **Output Layer:** Produces predictions

- **Key Terminology:**
  - **Dense/Fully Connected:** All neurons connected between layers
  - **Width:** Number of neurons in a layer
  - **Depth:** Number of layers in the network

- **Business Consideration:**
  - Deep networks can model complex relationships
  - Wider networks can capture more patterns

---

## Example: A Simple Neural Network in PyTorch

```python
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Non-linear activation function
        self.relu = nn.ReLU()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass through the network
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create a network with 10 inputs, 20 hidden neurons, and 2 outputs
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
```

---

## Feedforward Process

- **Forward Propagation:**
  - Data flows from input to output
  - Each layer applies weights, biases, and activation functions
  - Final layer produces prediction

- **Mathematical Operation:**
  - Layer 1: h₁ = f₁(W₁x + b₁)
  - Layer 2: h₂ = f₂(W₂h₁ + b₂)
  - Output: y = f₃(W₃h₂ + b₃)

- **Business Example:**
  - Credit scoring model processing multiple customer attributes
  - Each layer extracts higher-level patterns from raw data

---

## Learning Process: Backpropagation

- **Loss Function:** Measures prediction error
  - Binary Classification: Binary Cross Entropy
  - Multi-class: Categorical Cross Entropy
  - Regression: Mean Squared Error

- **Backpropagation:**
  - Calculate gradients (rate of change) for each weight
  - Determine how each parameter affects error
  - Update weights to minimize error

- **Gradient Descent:**
  - Small iterative updates to parameters
  - Learning rate controls update size
  - Balances learning speed and stability

---

## Training a Neural Network

```python
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(inputs)
    
    # Calculate loss
    loss = criterion(outputs, targets)
    
    # Backward pass and optimize
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```

---

## Practical Example: Sales Prediction

- **Business Problem:**
  - Predicting quarterly sales based on multiple factors
  - Input features: Economic indicators, marketing spend, seasonality, etc.
  - Output: Forecasted sales figures

- **Network Design:**
  - Input layer: 12 neurons (one per feature)
  - Hidden layers: 2 layers of 24 neurons each
  - Output layer: 1 neuron (sales prediction)

- **Business Impact:**
  - More accurate inventory planning
  - Optimized marketing spend allocation
  - Better cash flow management

---

## Handling Non-Linear Relationships

- **Problem:** Most business relationships are non-linear
  - Marketing spend vs. customer acquisition
  - Pricing vs. sales volume
  - Production volume vs. unit cost

- **Solution:** Deep neural networks with non-linear activations
  - Can model complex, non-linear patterns
  - Automatically learn appropriate transformations
  - Capture interactions between variables

- **Demo:** Fitting a non-linear function

---

## Example: Non-Linear Function Approximation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate data: y = x^2 with some noise
x = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = x**2 + 0.05 * torch.randn_like(x)

# Define a neural network model
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot results
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), model(x).detach().numpy(), 'r', label='NN Prediction')
plt.plot(x.numpy(), x.numpy()**2, 'g--', label='True Function (x²)')
plt.legend()
plt.show()
```

---

## Multi-Class Classification

- **Business Application:** Customer Segmentation
  - Classify customers into multiple segments
  - Enable targeted marketing strategies
  - Optimize product offerings by segment

- **Network Output Design:**
  - Output layer: One neuron per class
  - Softmax activation: Converts outputs to probabilities
  - Output sums to 1.0 across all classes

- **Evaluation Metrics:**
  - Accuracy: Overall correct classification rate
  - Confusion matrix: Detailed error analysis
  - F1-score: Balance of precision and recall

---

## Vanishing and Exploding Gradients

- **Problem:**
  - Deep networks can suffer from unstable gradients
  - Vanishing: Gradients become too small to learn
  - Exploding: Gradients become too large, causing instability

- **Business Impact:**
  - Failed model training
  - Extended development time
  - Unpredictable performance

- **Solutions:**
  - Better activation functions (ReLU)
  - Careful weight initialization
  - Batch normalization
  - Residual connections

---

## Transfer Learning: Leveraging Pre-trained Models

- **Concept:** Reuse knowledge from previously trained models

- **Business Advantages:**
  - Requires less training data
  - Reduces computation time and cost
  - Often produces better results
  - Faster time-to-market

- **Implementation:**
  - Start with pre-trained model (e.g., ImageNet models)
  - Freeze early layers (retain general features)
  - Retrain later layers for specific task
  - Fine-tune if necessary

---

## Learning Challenge: Credit Risk Prediction

**Exercise:** Design a neural network for predicting credit default risk:

1. What features (inputs) would you use?
2. How would you structure the network architecture?
3. What activation function would be appropriate for the output layer?
4. How would you evaluate the model's performance?
5. What business metrics would determine success?

**Discussion Questions:**
- How would you balance false positives (declining good customers) vs. false negatives (approving bad risks)?
- What regulatory considerations might apply to this model?
- How would you explain the model's decisions to stakeholders?

---

## Key Takeaways

- Neural networks learn hierarchical representations of data
- Non-linear activation functions enable complex pattern recognition
- Training involves forward propagation, loss calculation, and backpropagation
- Deep learning can model sophisticated non-linear relationships in business data
- Practical implementations require careful design and regularization
- Transfer learning offers efficient knowledge reuse for business applications
- Business context determines appropriate architecture and evaluation metrics 