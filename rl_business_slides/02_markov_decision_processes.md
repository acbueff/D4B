# Markov Decision Processes (MDPs)

---

## Introduction to MDPs

- **Definition:** Mathematical framework for modeling sequential decision-making problems
  - Agent interacts with environment over time
  - Actions influence both immediate rewards and future states
  - Goal is to maximize cumulative rewards

- **Formal MDP Definition:** 5-tuple (S, A, P, R, γ)
  - S: Set of states
  - A: Set of actions
  - P: State transition probability function P(s'|s,a)
  - R: Reward function R(s,a,s')
  - γ: Discount factor [0,1]

---

## The Markov Property

- **Markov Property:** The future depends only on the present, not the past
  - P(s<sub>t+1</sub> | s<sub>t</sub>, a<sub>t</sub>, s<sub>t-1</sub>, a<sub>t-1</sub>, ...) = P(s<sub>t+1</sub> | s<sub>t</sub>, a<sub>t</sub>)

- **Business Implications:**
  - Simplifies modeling by focusing only on current state
  - Reduces the complexity of the decision problem
  - May require creative state design to capture relevant history

- **Example:** Customer purchasing behavior
  - Markovian: Next purchase depends only on current customer segment
  - Non-Markovian: Next purchase depends on purchase history and seasonal factors

---

## State Transitions and Probability

- **Deterministic Environments:**
  - Each state-action pair leads to exactly one next state
  - Example: Basic inventory systems where ordering 10 units always increases inventory by 10

- **Stochastic Environments:**
  - State-action pairs lead to multiple possible next states with probabilities
  - Example: Marketing campaigns with variable response rates

- **Visualization: Transition Diagram**
  
  ```
  State A ---[Action 1, p=0.7]---> State B (Reward: +5)
         \--[Action 1, p=0.3]---> State C (Reward: +1)
          \-[Action 2, p=1.0]---> State D (Reward: +2)
  ```

---

## Example: Customer Relationship MDP

- **States:** Customer relationship stages
  - S = {Prospect, First-time, Active, At-risk, Churned, Returned}

- **Actions:** Marketing interventions
  - A = {Email campaign, Discount offer, Premium service, Winback campaign}

- **Transitions:** Probabilities of relationship changes
  - P(Active | First-time, Discount offer) = 0.6
  - P(At-risk | Active, No action) = 0.3

- **Rewards:** Business value of customer in each state
  - R(First-time → Active) = $100 (Customer acquisition value)
  - R(At-risk → Churned) = -$200 (Cost of churn)

---

## Discount Factor and Time Horizon

- **Discount Factor (γ):** Value between 0 and 1 that determines importance of future rewards
  - γ = 0: Myopic, only immediate rewards matter
  - γ = 1: All future rewards valued equally
  - 0 < γ < 1: Future rewards matter, but less than immediate rewards

- **Business Interpretation:**
  - Quarterly planning: Lower γ (0.7-0.8)
  - Strategic investment: Higher γ (0.9-0.99)
  - Reflects time preference for money and uncertainty

- **Example: Customer Lifetime Value (CLV)**
  - Low γ: Focuses on immediate purchase value
  - High γ: Properly values long-term customer relationships

---

## MDP Example Implementation

```python
import random

class RandomMDP:
    def __init__(self):
        self.states = ['A', 'B', 'C']  # Different market segments
        self.current_state = 'A'
    
    def reset(self):
        self.current_state = 'A'
        return self.current_state
    
    def step(self, action):
        # Action 0: Conservative marketing approach
        # Action 1: Aggressive marketing approach
        
        if action == 0:  # Conservative
            if self.current_state == 'A':
                # 70% stay in A, 30% move to B
                next_state = 'A' if random.random() < 0.7 else 'B'
                reward = 1  # Small but reliable reward
            elif self.current_state == 'B':
                # 60% stay in B, 40% move to C
                next_state = 'B' if random.random() < 0.6 else 'C'
                reward = 2  # Medium reward
            else:  # State C
                # 90% stay in C, 10% move to A
                next_state = 'C' if random.random() < 0.9 else 'A'
                reward = 3  # High reward
        else:  # Aggressive
            if self.current_state == 'A':
                # 40% stay in A, 60% move to either B or C
                r = random.random()
                next_state = 'A' if r < 0.4 else ('B' if r < 0.7 else 'C')
                reward = 0 if next_state == 'A' else (3 if next_state == 'B' else 5)
            elif self.current_state == 'B':
                # 30% stay in B, 30% to A, 40% to C
                r = random.random()
                next_state = 'B' if r < 0.3 else ('A' if r < 0.6 else 'C')
                reward = 2 if next_state == 'B' else (0 if next_state == 'A' else 4)
            else:  # State C
                # 50% stay in C, 50% back to A
                next_state = 'C' if random.random() < 0.5 else 'A'
                reward = 3 if next_state == 'C' else -1
        
        self.current_state = next_state
        return next_state, reward
```

---

## Solving MDPs: Value Iteration

- **Goal:** Find the optimal policy π* that maximizes expected future rewards

- **Value Function:** V(s) = Expected sum of discounted rewards starting from state s
  - V<sup>*</sup>(s) = max<sub>a</sub> [R(s,a) + γ∑<sub>s'</sub>P(s'|s,a)V<sup>*</sup>(s')]

- **Value Iteration Algorithm:**
  1. Initialize V(s) = 0 for all states
  2. Repeat until convergence:
     - For each state s, update V(s) = max<sub>a</sub> [R(s,a) + γ∑<sub>s'</sub>P(s'|s,a)V(s')]
  3. Extract policy: π*(s) = argmax<sub>a</sub> [R(s,a) + γ∑<sub>s'</sub>P(s'|s,a)V(s')]

---

## Business Applications of MDPs

- **Dynamic Pricing:**
  - States: Inventory levels, competitor prices, market demand
  - Actions: Price points, discount levels
  - Goal: Maximize revenue while managing inventory

- **Product Development:**
  - States: Development phases, market conditions
  - Actions: Investment levels, feature prioritization
  - Goal: Optimize product-market fit and ROI

- **Financial Portfolio Management:**
  - States: Market conditions, portfolio composition
  - Actions: Buy, sell, hold decisions
  - Goal: Maximize risk-adjusted returns

---

## Limitations and Practical Considerations

- **State Space Explosion:**
  - Real business problems may have enormous state spaces
  - Solution: Function approximation, state aggregation

- **Partial Observability:**
  - Many business environments aren't fully observable
  - Solution: POMDPs (Partially Observable MDPs)

- **Non-Stationarity:**
  - Business environments change over time
  - Solution: Continuous learning, adaptive models

- **Data Requirements:**
  - Transition probabilities often unknown initially
  - Solution: Reinforcement learning (learning from experience)

---

## Learning Challenge: MDP Formulation

**Exercise:** Consider a subscription business (e.g., streaming service):

1. Define the state space in terms of customer attributes
2. Define the action space (marketing/retention activities)
3. Sketch out potential transition probabilities
4. Define an appropriate reward function
5. What value of discount factor would be appropriate?

**Discussion Questions:**
- How would changing the reward function change the optimal policy?
- What historical data would you need to estimate the transition probabilities?
- How could you incorporate business constraints (e.g., marketing budget)?

---

## Key Takeaways

- MDPs provide a mathematical framework for sequential decision-making
- The Markov property simplifies modeling but may require careful state design
- Discount factor balances short-term and long-term rewards
- Solving MDPs yields optimal policies for maximizing cumulative rewards
- Business applications span marketing, pricing, product development, and finance
- Practical challenges include state space explosion and unknown transition probabilities 