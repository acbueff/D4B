"""
Business Data Fundamentals for AI Applications
===========================================

This script demonstrates key concepts about data and datasets in a business context,
using a retail business scenario. It's designed for business professionals learning
about AI and data science.

Author: Claude
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    """Helper function to print formatted section titles"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80 + "\n")

#=============================================================================
# Section 1: Understanding Business Data Types
#=============================================================================

print_section("SECTION 1: UNDERSTANDING BUSINESS DATA TYPES")

# Create example data for a retail business
print("Creating sample retail business data...")

# Structured data example: Customer purchase records
structured_data = pd.DataFrame({
    'customer_id': [1001, 1002, 1003, 1004],
    'purchase_amount': [150.50, 200.75, 75.25, 300.00],
    'items_bought': [3, 4, 1, 5],
    'is_member': [True, False, True, True]
})

print("\nStructured Data Example (Customer Purchase Records):")
print(structured_data)

# Semi-structured data example: Customer feedback in JSON format
semi_structured_data = [
    {'customer_id': 1001, 'rating': 4, 'comments': 'Good service', 'tags': ['friendly', 'fast']},
    {'customer_id': 1002, 'rating': 5, 'comments': 'Excellent products', 'tags': ['quality', 'variety']}
]

print("\nSemi-structured Data Example (Customer Feedback):")
for review in semi_structured_data:
    print(review)

# Unstructured data example: Customer service notes
unstructured_data = """
Customer called regarding product return.
Very satisfied with service but product didn't meet expectations.
Offered 10% discount on next purchase.
"""

print("\nUnstructured Data Example (Customer Service Notes):")
print(unstructured_data)

# Reflection Questions
print("\nReflection Questions:")
print("1. Which type of data would be most valuable for predicting customer churn?")
print("2. How might combining different data types improve business insights?")

#=============================================================================
# Section 2: Data Quality Assessment and Preprocessing
#=============================================================================

print_section("SECTION 2: DATA QUALITY ASSESSMENT AND PREPROCESSING")

# Create sample sales data with quality issues
print("Creating sample sales data with quality issues...")

sales_data = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'product_id': ['A001', 'B002', 'A001', 'C003', 'B002'],
    'quantity': [10, -5, 15, 0, 8],
    'price': [99.99, 149.99, 99.99, 199.99, 150.00],
    'customer_segment': ['Premium', 'Basic', 'Unknown', 'Premium', 'Basic']
})

print("\nOriginal Sales Data:")
print(sales_data)

# Check for data quality issues
print("\nData Quality Assessment:")
print(f"1. Missing values: {sales_data.isnull().sum().sum()}")
print(f"2. Negative values in quantity: {(sales_data['quantity'] < 0).sum()}")
print(f"3. Unknown categories: {(sales_data['customer_segment'] == 'Unknown').sum()}")

# Clean the data
cleaned_data = sales_data.copy()
# Fix negative quantities
cleaned_data.loc[cleaned_data['quantity'] < 0, 'quantity'] = 0
# Replace unknown customer segment
cleaned_data['customer_segment'] = cleaned_data['customer_segment'].replace('Unknown', 'Basic')

print("\nCleaned Sales Data:")
print(cleaned_data)

# Reflection Questions
print("\nReflection Questions:")
print("1. What business impact could negative quantity values have on analysis?")
print("2. How would you handle 'Unknown' values in a real business scenario?")

#=============================================================================
# Section 3: Handling Missing Data
#=============================================================================

print_section("SECTION 3: HANDLING MISSING DATA")

# Create customer data with missing values
print("Creating customer data with missing values...")

customer_data = pd.DataFrame({
    'customer_id': range(1001, 1006),
    'age': [35, np.nan, 45, 28, np.nan],
    'total_purchases': [1200, 800, np.nan, 1500, 950],
    'loyalty_years': [3, 2, 4, np.nan, 1]
})

print("\nCustomer Data with Missing Values:")
print(customer_data)

# Analyze missing data
print("\nMissing Data Analysis:")
print(customer_data.isnull().sum())

# Handle missing data using different methods
# 1. Mean imputation for numerical data
customer_data['age'].fillna(customer_data['age'].mean(), inplace=True)
customer_data['total_purchases'].fillna(customer_data['total_purchases'].mean(), inplace=True)
customer_data['loyalty_years'].fillna(customer_data['loyalty_years'].mean(), inplace=True)

print("\nCustomer Data After Handling Missing Values:")
print(customer_data)

# Reflection Questions
print("\nReflection Questions:")
print("1. What are the pros and cons of using mean imputation for missing values?")
print("2. How might missing data affect customer segmentation analysis?")

#=============================================================================
# Section 4: Dataset Splitting for Business Analytics
#=============================================================================

print_section("SECTION 4: DATASET SPLITTING FOR BUSINESS ANALYTICS")

# Create sample customer churn data
print("Creating sample customer churn data...")

np.random.seed(42)  # For reproducibility
n_customers = 1000

churn_data = pd.DataFrame({
    'usage_frequency': np.random.normal(10, 3, n_customers),
    'support_calls': np.random.poisson(2, n_customers),
    'tenure_months': np.random.randint(1, 60, n_customers),
    'monthly_charges': np.random.uniform(50, 150, n_customers),
    'churned': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
})

print("\nSample of Customer Churn Data:")
print(churn_data.head())

# Split the data
X = churn_data.drop('churned', axis=1)
y = churn_data['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDataset Splitting Results:")
print(f"Training set size: {len(X_train)} customers")
print(f"Test set size: {len(X_test)} customers")
print(f"Churn rate in training set: {y_train.mean():.2%}")
print(f"Churn rate in test set: {y_test.mean():.2%}")

# Reflection Questions
print("\nReflection Questions:")
print("1. Why is it important to maintain similar churn rates in train and test sets?")
print("2. How would you handle highly imbalanced classes in churn prediction?")

#=============================================================================
# Section 5: Feature Engineering for Business Insights
#=============================================================================

print_section("SECTION 5: FEATURE ENGINEERING FOR BUSINESS INSIGHTS")

# Create sample transaction data
print("Creating sample transaction data...")

transactions = pd.DataFrame({
    'customer_id': np.repeat(range(1001, 1004), 5),
    'transaction_date': pd.date_range(start='2024-01-01', periods=15, freq='D'),
    'amount': np.random.uniform(50, 500, 15)
})

print("\nOriginal Transaction Data:")
print(transactions)

# Feature engineering examples
# 1. Calculate customer-level features
customer_features = transactions.groupby('customer_id').agg({
    'amount': ['mean', 'sum', 'count'],
    'transaction_date': ['min', 'max']
}).reset_index()

# Flatten column names
customer_features.columns = ['customer_id', 'avg_amount', 'total_spend', 
                           'transaction_count', 'first_purchase', 'last_purchase']

# 2. Add derived features
customer_features['customer_age_days'] = (
    customer_features['last_purchase'] - customer_features['first_purchase']
).dt.days

customer_features['avg_transaction_frequency'] = (
    customer_features['transaction_count'] / 
    customer_features['customer_age_days']
)

print("\nEngineered Customer Features:")
print(customer_features)

# Reflection Questions
print("\nReflection Questions:")
print("1. Which engineered features might be most predictive of customer value?")
print("2. What additional features could be created from this data?")

#=============================================================================
# Section 6: Database Fundamentals for Business
#=============================================================================

print_section("SECTION 6: DATABASE FUNDAMENTALS FOR BUSINESS")

print("""
Understanding Databases in Business
---------------------------------
A database is a structured collection of related data that represents some aspect
of an organization's operations. Modern businesses use Database Management Systems
(DBMS) to store, retrieve, and manage their data efficiently.
""")

# Create example tables for a simple retail database
print("Creating sample retail database tables...")

# Products table
products = pd.DataFrame({
    'product_id': ['P001', 'P002', 'P003', 'P004'],
    'name': ['Laptop', 'Smartphone', 'Tablet', 'Headphones'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Accessories'],
    'price': [1200.00, 800.00, 500.00, 100.00],
    'stock': [50, 100, 75, 200]
})

# Customers table
customers = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003'],
    'name': ['John Smith', 'Jane Doe', 'Bob Wilson'],
    'email': ['john@email.com', 'jane@email.com', 'bob@email.com'],
    'join_date': ['2024-01-01', '2024-01-15', '2024-02-01']
})

# Orders table
orders = pd.DataFrame({
    'order_id': ['O001', 'O002', 'O003', 'O004'],
    'customer_id': ['C001', 'C002', 'C001', 'C003'],
    'product_id': ['P001', 'P002', 'P003', 'P001'],
    'quantity': [1, 2, 1, 1],
    'order_date': ['2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04']
})

print("\nDatabase Tables:")
print("\nProducts Table:")
print(products)
print("\nCustomers Table:")
print(customers)
print("\nOrders Table:")
print(orders)

# Demonstrate basic SQL-like operations using pandas
print("\nDatabase Operations Examples:")

# 1. Simple SELECT (Filtering)
print("\n1. Find all products with price > $500:")
expensive_products = products[products['price'] > 500]
print(expensive_products)

# 2. JOIN operation (Combining tables)
print("\n2. Orders with customer names:")
orders_with_customers = pd.merge(
    orders, 
    customers[['customer_id', 'name']], 
    on='customer_id'
)
print(orders_with_customers)

# 3. Aggregation (GROUP BY)
print("\n3. Total orders by customer:")
customer_orders = orders.groupby('customer_id').agg({
    'order_id': 'count',
    'quantity': 'sum'
}).reset_index()
print(customer_orders)

# Create visualizations
plt.figure(figsize=(15, 5))

# 1. Product Prices Bar Chart
plt.subplot(131)
plt.bar(products['name'], products['price'])
plt.title('Product Prices')
plt.xticks(rotation=45)
plt.ylabel('Price ($)')

# 2. Stock Levels Pie Chart
plt.subplot(132)
plt.pie(products['stock'], labels=products['name'], autopct='%1.1f%%')
plt.title('Stock Distribution')

# 3. Orders Timeline
plt.subplot(133)
order_counts = pd.to_datetime(orders['order_date']).value_counts().sort_index()
plt.plot(order_counts.index, order_counts.values, marker='o')
plt.title('Orders Timeline')
plt.xticks(rotation=45)
plt.ylabel('Number of Orders')

plt.tight_layout()
plt.show()

# Database Schema Visualization (using text-based representation)
print("\nDatabase Schema:")
print("""
Products Table                 Customers Table
+-------------+-------+       +-------------+-------+
| product_id  | PK    |       | customer_id | PK    |
| name        |       |       | name        |       |
| category    |       |       | email       |       |
| price       |       |       | join_date   |       |
| stock       |       |       +-------------+-------+
+-------------+-------+               ↑
      ↑                              |
      |                              |
      |         Orders Table         |
      |    +-------------+-------+   |
      +----| product_id  | FK    |   |
           | customer_id | FK    |---+
           | order_id    | PK    |
           | quantity    |       |
           | order_date  |       |
           +-------------+-------+
""")

# Reflection Questions
print("\nReflection Questions:")
print("1. How does a relational database structure help maintain data integrity?")
print("2. What business insights can be derived from joining different tables?")
print("3. How might this database schema evolve as the business grows?")

#=============================================================================
# Final Coding Challenge: Customer Segmentation
#=============================================================================

print_section("FINAL CODING CHALLENGE: CUSTOMER SEGMENTATION")

print("""
Challenge: Create a Basic Customer Segmentation System
---------------------------------------------------

Your task is to create a simple customer segmentation system based on:
1. Total spending
2. Transaction frequency
3. Average transaction amount

Requirements:
1. Use the provided transaction data
2. Create customer-level features
3. Segment customers into 3 tiers: 'High Value', 'Medium Value', 'Low Value'
4. Print the number of customers in each segment
5. Calculate average metrics for each segment

Success Criteria:
1. Code runs without errors
2. All required features are calculated correctly
3. Segments are created using logical thresholds
4. Output includes segment sizes and average metrics
5. Code includes clear comments explaining the process

Template code is provided below. Complete the missing sections.
""")

def segment_customers(transactions):
    '''
    Segments customers based on their transaction patterns.
    
    Parameters:
        transactions: DataFrame with columns [customer_id, transaction_date, amount]
    
    Returns:
        DataFrame with customer segments and metrics
    '''
    # Step 1: Calculate customer-level metrics
    # TODO: Group by customer_id and calculate:
    # - total_spend
    # - transaction_count
    # - avg_transaction
    
    # Step 2: Create segments
    # TODO: Add segment column based on total_spend:
    # - High Value: Top 20%
    # - Medium Value: Middle 40%
    # - Low Value: Bottom 40%
    
    # Step 3: Calculate segment metrics
    # TODO: Calculate average metrics for each segment
    
    return customer_segments

# Sample solution (hidden from students)
def solution_segment_customers(transactions):
    """Solution to the customer segmentation challenge."""
    # Calculate customer metrics
    customer_metrics = transactions.groupby('customer_id').agg({
        'amount': ['sum', 'count', 'mean']
    }).reset_index()
    
    customer_metrics.columns = ['customer_id', 'total_spend', 
                              'transaction_count', 'avg_transaction']
    
    # Create segments based on total spend
    spend_threshold_high = customer_metrics['total_spend'].quantile(0.8)
    spend_threshold_medium = customer_metrics['total_spend'].quantile(0.4)
    
    customer_metrics['segment'] = 'Low Value'
    customer_metrics.loc[customer_metrics['total_spend'] >= spend_threshold_medium, 
                        'segment'] = 'Medium Value'
    customer_metrics.loc[customer_metrics['total_spend'] >= spend_threshold_high, 
                        'segment'] = 'High Value'
    
    # Calculate segment metrics
    segment_metrics = customer_metrics.groupby('segment').agg({
        'customer_id': 'count',
        'total_spend': 'mean',
        'transaction_count': 'mean',
        'avg_transaction': 'mean'
    }).round(2)
    
    segment_metrics.columns = ['Customer Count', 'Avg Total Spend', 
                             'Avg Transaction Count', 'Avg Transaction Amount']
    
    return segment_metrics

# Test the solution
test_transactions = pd.DataFrame({
    'customer_id': np.repeat(range(1001, 1021), 5),
    'transaction_date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'amount': np.random.uniform(50, 500, 100)
})

print("\nSample Solution Output:")
print(solution_segment_customers(test_transactions))

print("\nChallenge Complete!") 