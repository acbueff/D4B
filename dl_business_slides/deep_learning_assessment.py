import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

"""
Deep Learning Concepts Demo and Assessment
----------------------------------------

This script demonstrates key deep learning concepts through a business-relevant
customer churn prediction task. Throughout the code, you'll find:

1. Data Preparation and Feature Engineering
2. Neural Network Architecture Design
3. Training Process and Optimization
4. Model Evaluation and Business Impact Assessment

Reflection Questions are marked with [R] throughout the code.
Learning Objectives are marked with [L].
Business Implications are marked with [B].

[R] Before starting: What factors do you think contribute most to customer churn?
[B] How would you quantify the business cost of false negatives vs false positives?
"""

def generate_customer_data(n_samples=1000):
    """
    [L] Data Generation and Feature Engineering
    - Demonstrates how real-world business metrics become model features
    - Shows relationships between different customer attributes
    
    [R] Consider: Why might longer tenure correlate with lower churn?
    [R] What other features might be valuable for churn prediction?
    """
    # Generate features
    tenure = np.random.randint(1, 72, n_samples)  # 1-72 months
    monthly_charges = np.random.uniform(30, 150, n_samples)  # $30-$150
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
    support_calls = np.random.poisson(2, n_samples)  # Average 2 calls
    usage_score = np.random.uniform(0, 100, n_samples)  # 0-100 usage score
    
    # Combine features
    X = np.column_stack([tenure, monthly_charges, total_charges, support_calls, usage_score])
    
    # Generate churn labels based on a rule
    churn_score = (
        -0.1 * tenure +  # Longer tenure = less likely to churn
        0.3 * (monthly_charges / 50) +  # Higher charges = more likely to churn
        0.2 * (support_calls / 2) +  # More support calls = more likely to churn
        -0.4 * (usage_score / 50)  # Higher usage = less likely to churn
    )
    
    churn_score += np.random.normal(0, 0.1, n_samples)
    y = (churn_score > 0).astype(np.int64)
    
    return X, y

def plot_feature_relationships(X_train, y_train):
    """
    [L] Data Visualization and Pattern Recognition
    - Shows distribution of features for churned vs retained customers
    - Demonstrates the importance of data exploration before modeling
    
    [R] What patterns do you observe in the visualizations?
    [R] How might these patterns inform business strategies?
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    feature_names = ['Tenure', 'Monthly Charges', 'Support Calls', 'Usage Score']
    feature_indices = [0, 1, 3, 4]
    
    for i, (name, idx) in enumerate(zip(feature_names, feature_indices)):
        churned = X_train[y_train == 1, idx]
        stayed = X_train[y_train == 0, idx]
        
        axes[i].hist([stayed, churned], bins=20, label=['Stayed', 'Churned'], alpha=0.6)
        axes[i].set_title(f'{name} Distribution by Churn Status')
        axes[i].set_xlabel(name)
        axes[i].set_ylabel('Count')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

class ChurnPredictor(nn.Module):
    """
    [L] Neural Network Architecture
    - Demonstrates basic feed-forward neural network design
    - Shows how activation functions introduce non-linearity
    
    [R] Why do we need non-linear activation functions?
    [R] How does the network architecture relate to the complexity of patterns it can learn?
    """
    def __init__(self, input_size=5):
        super(ChurnPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # [L] Forward Propagation
        x = self.relu(self.layer1(x))  # ReLU helps with vanishing gradient
        x = self.relu(self.layer2(x))  # Multiple layers capture complex patterns
        x = self.sigmoid(self.layer3(x))  # Sigmoid squashes output to [0,1]
        return x

def plot_confusion_matrix(y_true, y_pred):
    """
    [L] Model Evaluation Visualization
    [B] Business Impact Analysis
    
    Shows confusion matrix with business implications:
    - False Positives: Cost of unnecessary retention efforts
    - False Negatives: Cost of lost customers
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Calculate business metrics
    retention_rate = cm[0,0] / (cm[0,0] + cm[0,1])
    detection_rate = cm[1,1] / (cm[1,0] + cm[1,1])
    print(f"\nBusiness Metrics:")
    print(f"Customer Retention Rate: {retention_rate:.2%}")
    print(f"Churn Detection Rate: {detection_rate:.2%}")
    
    # [R] How would you adjust the model threshold based on these metrics?
    # [R] What's more costly: missing churners or falsely flagging loyal customers?

def train_model(model, X_train, y_train, X_test, y_test, criterion, optimizer, epochs=100):
    """
    [L] Model Training Process
    - Demonstrates gradient descent and backpropagation
    - Shows importance of monitoring training progress
    
    [R] Why do we use both training and test sets?
    [R] What patterns in the learning curves suggest overfitting?
    """
    train_losses = []
    test_metrics = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()  # Backpropagation
        optimizer.step()  # Gradient descent
        train_losses.append(loss.item())
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_preds = (test_outputs >= 0.5).float()
                accuracy = accuracy_score(y_test, test_preds)
                f1 = f1_score(y_test, test_preds)
                test_metrics.append((epoch, accuracy, f1))
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
                      f'Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
    
    return train_losses, test_metrics

class ImprovedChurnPredictor(nn.Module):
    """
    [L] Advanced Architecture Features
    - Demonstrates regularization techniques
    - Shows how to prevent overfitting
    
    [R] How does dropout help prevent overfitting?
    [R] Why use different layer sizes in the architecture?
    """
    def __init__(self, input_size=5):
        super(ImprovedChurnPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 8)
        self.layer4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.batch_norm1(self.dropout(self.relu(self.layer1(x))))
        x = self.batch_norm2(self.dropout(self.relu(self.layer2(x))))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x

def plot_training_progress(train_losses, test_metrics):
    """
    [L] Visualizing the Training Process
    - Shows how loss and metrics change during training
    - Helps identify overfitting/underfitting
    
    [R] What does the convergence of the loss curve tell us?
    [R] How do we know if we've trained for enough epochs?
    """
    epochs, accuracies, f1_scores = zip(*test_metrics)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Plot test metrics
    ax2.plot(epochs, accuracies, label='Accuracy')
    ax2.plot(epochs, f1_scores, label='F1 Score')
    ax2.set_title('Test Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    [L] Model Evaluation
    - Demonstrates various metrics for model assessment
    - Shows how to interpret model performance
    
    [R] Which metric is most important for our business case?
    [B] How do these metrics translate to business value?
    """
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_preds = (test_outputs >= 0.5).float()
        
        accuracy = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        
        print("\nFinal Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, test_preds))
        
        # Check if model passes assessment criteria
        passes_accuracy = accuracy > 0.85
        passes_f1 = f1 > 0.80
        
        print("\nAssessment Results:")
        print(f"Accuracy > 85%: {'✓' if passes_accuracy else '✗'}")
        print(f"F1 Score > 0.80: {'✓' if passes_f1 else '✗'}")
        print(f"Overall: {'PASS' if (passes_accuracy and passes_f1) else 'FAIL'}")
        
        return passes_accuracy and passes_f1

def main():
    """
    Main execution flow with business context and learning objectives
    """
    print(__doc__)  # Print the module docstring with learning objectives
    
    # Generate and prepare data
    X, y = generate_customer_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    print("\nDataset Information:")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Churn rate in training set: {y_train.mean():.2%}")
    
    # Data visualization
    print("\n[L] Exploring Feature Relationships")
    plot_feature_relationships(X_train, y_train)
    input("\nPress Enter after analyzing the visualizations...")
    
    # Train base model
    print("\n[L] Training Base Model")
    model = ChurnPredictor()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses, test_metrics = train_model(model, X_train_tensor, y_train_tensor, 
                                           X_test_tensor, y_test_tensor,
                                           criterion, optimizer)
    
    print("\n[L] Analyzing Model Performance")
    plot_training_progress(train_losses, test_metrics)
    base_passed = evaluate_model(model, X_test_tensor, y_test_tensor)
    
    # Confusion matrix for business impact
    with torch.no_grad():
        base_preds = (model(X_test_tensor) >= 0.5).float()
    plot_confusion_matrix(y_test_tensor, base_preds)
    
    # Train improved model
    print("\n[L] Training Improved Model with Advanced Features")
    improved_model = ImprovedChurnPredictor()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(improved_model.parameters(), lr=0.005)
    
    train_losses, test_metrics = train_model(improved_model, X_train_tensor, y_train_tensor, 
                                           X_test_tensor, y_test_tensor,
                                           criterion, optimizer, epochs=150)
    
    print("\n[L] Comparing Model Performances")
    plot_training_progress(train_losses, test_metrics)
    improved_passed = evaluate_model(improved_model, X_test_tensor, y_test_tensor)
    
    # Final business impact analysis
    with torch.no_grad():
        improved_preds = (improved_model(X_test_tensor) >= 0.5).float()
    plot_confusion_matrix(y_test_tensor, improved_preds)
    
    print("\nReflection Questions:")
    print("[R] What improvements in the advanced model led to better performance?")
    print("[R] How would you implement this model in a production environment?")
    print("[R] What additional data would make the model more effective?")
    print("[B] How would you quantify the ROI of this churn prediction system?")

if __name__ == "__main__":
    main() 