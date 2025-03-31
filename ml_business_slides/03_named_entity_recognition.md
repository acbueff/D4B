# Named Entity Recognition (NER)

## Business Context: JPMorgan's COIN System

JPMorgan's Contract Intelligence (COIN) system:
- Analyzes 12,000+ commercial credit agreements annually
- Extracts key data points and contract terms in seconds
- Reduces 360,000 hours of lawyer/loan officer work
- Minimizes human error in contract interpretation

## What is Named Entity Recognition?

NER identifies and classifies specific entities in text:
- People: "John Smith signed the agreement"
- Organizations: "Google acquired DeepMind in 2014"
- Dates: "The payment is due on January 15th, 2023"
- Money: "The loan amount of $250,000 will be disbursed"
- Locations: "The property at 123 Main Street, New York"

## Notebook Demonstration

We'll explore NER using spaCy's pre-trained model:
- Analyze a sample loan agreement
- Identify entities like dates, monetary values, and organizations
- Extract specific information such as loan amount and term
- Build a simple rule-based extractor for a specific contract element

```python
# Sample code (from the notebook)
import spacy

# Load spaCy's pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Sample text from a loan agreement
text = """Borrower agrees to pay Lender the sum of $50,000 with 
an annual interest rate of 5.25% over a period of 10 years, 
with payments due on the 1st of each month beginning January 1, 2023."""

# Process the text
doc = nlp(text)

# Display entities
for ent in doc.ents:
    print(f"Entity: {ent.text}, Type: {ent.label_}")

# Extract loan details using regex or pattern matching
import re
loan_amount = re.search(r'\$([0-9,]+)', text).group(0)
loan_term = re.search(r'(\d+) years', text).group(0)
print(f"Loan amount: {loan_amount}, Term: {loan_term}")
```

## Interpreting Results

In our example, spaCy identifies:
- "$50,000" as MONEY
- "5.25%" as PERCENT
- "10 years" as TIME
- "January 1, 2023" as DATE

This enables automatic extraction of:
- Loan principal: $50,000
- Interest rate: 5.25%
- Loan term: 10 years
- First payment date: January 1, 2023

## Learning Challenge

Analyze this lease agreement snippet:
"Tenant shall pay Landlord a security deposit of $2,500 upon signing this lease, which expires on December 31, 2023."

1. What entities does spaCy identify?
2. Create a simple rule to extract the security deposit amount
3. How might you extract the lease expiration date?

## Real-world Implementation

- Financial institutions combine NER with custom entity types
- Advanced systems use transformer-based models for better accuracy
- Rule-based systems complement ML for domain-specific extraction
- Continuous training with new documents improves performance 