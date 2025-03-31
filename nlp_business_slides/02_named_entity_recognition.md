# Named Entity Recognition for Contract Analysis

---

## Understanding Named Entity Recognition (NER)

- **What is Named Entity Recognition?**
  - NLP technique that identifies and classifies named entities in text
  - Detects proper nouns and specific information items
  - Tags entities with predefined categories (Person, Organization, Date, Money, etc.)

- **Business Application: JPMorgan's COIN System**
  - Contract Intelligence (COIN) system analyzes legal documents
  - Processes 12,000+ commercial credit agreements annually
  - Extracts key data points and clauses from contracts
  - Saves 360,000 hours of lawyer/loan officer work

---

## Types of Named Entities

- **Standard Entity Types:**
  - PERSON: Names of people
  - ORG: Companies, institutions, agencies
  - GPE/LOC: Countries, cities, locations
  - DATE/TIME: Dates and times
  - MONEY: Monetary values
  - PERCENT: Percentage values

- **Domain-Specific Entities (in Legal Contracts):**
  - PARTY: Contracting parties
  - TERM: Contract duration
  - OBLIGATION: Legal requirements
  - CONDITION: Contingent clauses
  - PENALTY: Consequences for breach

---

## NER Approaches

- **Rule-based Systems:**
  - Pattern matching with regular expressions
  - Dictionary lookups and gazetteers
  - Grammatical rules

- **Statistical Machine Learning:**
  - Conditional Random Fields (CRFs)
  - Hidden Markov Models

- **Deep Learning:**
  - Bidirectional LSTMs with CRF layer
  - Transformer-based models (BERT, RoBERTa)

---

## Practical Implementation with spaCy

```python
# Example code using spaCy for contract analysis
import spacy

# Load pre-trained English model
nlp = spacy.load("en_core_web_lg")

# Sample contract text
contract_text = """The Borrower shall pay JPMorgan Chase Bank 
a sum of $5,000 on the 1st of each month over a loan term of 5 years
starting from January 15, 2023."""

# Process the text
doc = nlp(contract_text)

# Extract and display entities
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
    
# Custom pattern matching for contract-specific terms
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
loan_term_pattern = [{"LOWER": "loan"}, {"LOWER": "term"}, 
                     {"OP": "?"}, {"LIKE_NUM": True}, {"LOWER": "years"}]
matcher.add("LOAN_TERM", [loan_term_pattern])
```

---

## Entity Extraction from Contracts

**Sample Output from NER:**

| Entity | Type | Business Significance |
|--------|------|----------------------|
| JPMorgan Chase Bank | ORG | Lender party |
| $5,000 | MONEY | Payment amount |
| 1st of each month | DATE | Payment schedule |
| 5 years | DURATION | Loan term |
| January 15, 2023 | DATE | Start date |

**Benefits for Contract Analysis:**
- Automatic extraction of critical terms
- Standardization of contract information
- Risk identification and compliance monitoring

---

## JPMorgan's COIN Implementation

- **Evolution from Manual to Automated:**
  - Previously: Lawyers spent 360,000 hours reviewing contracts
  - Now: Machine extracts key data points in seconds
  
- **Technical Architecture:**
  - Custom-trained NER models for financial documents
  - Combination of rules and machine learning
  - Integration with workflow systems

- **Business Impact:**
  - 80% reduction in time spent on routine contract review
  - Decreased errors in interpretation
  - Freed legal resources for higher-value work

---

## Beyond Basic NER: Relation Extraction

- **Entity Recognition → Relationship Understanding:**
  - Not just identifying parties, but their roles
  - Connecting payment amounts to schedules
  - Linking conditions to consequences

- **Example Relationships in Contracts:**
  - Party-to-Obligation relationships
  - Payment-to-Timeline connections
  - Condition-to-Action dependencies

```
"If [PARTY: the Borrower] fails to make [PAYMENT: any scheduled payment], 
[PENALTY: a late fee of 2%] shall apply after [TIME: 10 business days]."
```

---

## Challenges in Legal NER

- **Domain-Specific Language:**
  - Legal terminology differs from everyday language
  - Context-dependent meanings

- **Document Structure:**
  - Complex nested clauses
  - References to other sections
  - Formatting variations

- **Ambiguity:**
  - "Party" as a celebration vs. contract participant
  - Dates with different formats (01/02/03 - Jan 2, 2003 or Feb 1, 2003?)
  - References using pronouns and anaphora

---

## Learning Challenge

**Exercise:** Identify entities in this contract snippet:

"Tenant agrees to pay Landlord a security deposit of $2,500 before May 1, 2023. The lease term shall be 24 months commencing on June 1, 2023."

**Discussion Questions:**
- What entities would be most valuable to extract automatically?
- How could a rule-based approach complement the ML model for this text?
- What relationships between entities are important to capture?

---

## Key Takeaways

- NER enables automatic extraction of critical information from contracts
- JPMorgan's COIN system demonstrates massive efficiency gains (360,000 hours saved)
- Practical implementation combines pre-trained models with domain customization
- Extracting relationships between entities provides deeper contract understanding
- Domain adaptation is crucial for specialized fields like legal document processing 