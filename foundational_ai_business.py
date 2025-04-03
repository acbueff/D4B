# Foundational AI for Business Applications
# =======================================
#
# This educational script demonstrates key concepts from the Foundational AI course
# with practical business applications. Each section includes:
# 1. Topic explanation
# 2. Code demonstration
# 3. Visual example
# 4. Reflective questions

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import heapq
import random

# Set up plotting
plt.style.use('default')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

def print_header():
    print("Foundational AI for Business")
    print("===========================")
    print("""
Welcome to the Foundational AI for Business tutorial. This script will teach you
key AI concepts through practical business examples. We'll cover:

1. Problem Formulation: How to translate business problems into AI terms
2. Search Algorithms: Finding optimal solutions in complex business scenarios
3. Local Search: Optimizing business resources and processes
4. Adversarial Search: Strategic decision making in competitive markets

Each topic includes code demonstrations, visualizations, and reflection questions
to help you understand how these concepts apply to real business situations.
""")

# Part 1: Problem Formulation
# ==========================

def explain_problem_formulation():
    print("""
TOPIC 1: PROBLEM FORMULATION IN AI
=================================

In AI, the first step is translating a business problem into formal terms:
- State Space: All possible situations we might encounter
- Actions: What we can do in each situation
- Transition Model: How our actions change the situation
- Goal State: What we're trying to achieve
- Cost Function: How to measure the cost of actions

Business Example: Delivery Route Optimization
------------------------------------------
We'll demonstrate this using a delivery routing problem where a company needs to:
- Deliver to multiple store locations
- Minimize total distance traveled
- Track costs between locations

This example shows how to:
1. Define the problem space (store locations)
2. Calculate costs (distances between stores)
3. Visualize the problem for better understanding
""")

class BusinessRoute:
    """
    Demonstrates problem formulation using a business delivery routing problem.
    This shows how to formulate a real business problem in AI terms.
    """
    def __init__(self):
        # Example business locations with (x, y) coordinates
        self.locations = {
            'Warehouse': (0, 0),
            'Store1': (2, 3),
            'Store2': (-1, 4),
            'Store3': (3, -2),
            'Store4': (-2, -3)
        }
        self.costs = self._calculate_costs()
    
    def _calculate_costs(self):
        """Calculate Euclidean distances between all locations"""
        costs = {}
        for loc1 in self.locations:
            for loc2 in self.locations:
                if loc1 != loc2:
                    x1, y1 = self.locations[loc1]
                    x2, y2 = self.locations[loc2]
                    costs[(loc1, loc2)] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return costs
    
    def visualize_problem(self):
        """Visualize the delivery locations and connections"""
        plt.figure(figsize=(10, 10))
        # Plot locations
        for loc, (x, y) in self.locations.items():
            plt.plot(x, y, 'bo', markersize=15)
            plt.text(x+0.1, y+0.1, loc, fontsize=12)
        
        # Plot connections with costs
        for (loc1, loc2), cost in self.costs.items():
            x1, y1 = self.locations[loc1]
            x2, y2 = self.locations[loc2]
            plt.plot([x1, x2], [y1, y2], 'k--', alpha=0.2)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            plt.text(mid_x, mid_y, f'{cost:.1f}', alpha=0.5)
        
        plt.title('Delivery Network with Distances')
        plt.grid(True)
        plt.show()

def reflection_problem_formulation():
    print("""
Reflection Questions - Problem Formulation:
-----------------------------------------
1. How would you modify this formulation for a different business problem,
   such as resource allocation or inventory management?
2. What additional constraints might real-world delivery problems have that
   aren't captured in this simple model?
3. How could you adapt the cost calculation to include factors beyond distance,
   such as time of day or delivery priority?
""")

# Part 2: Search Algorithms
# =======================

def explain_search_algorithms():
    print("""
TOPIC 2: SEARCH ALGORITHMS
=========================

Search algorithms help find optimal solutions in a structured way:
- Breadth-First Search: Explores all possibilities level by level
- Depth-First Search: Explores one path fully before backtracking
- A* Search: Uses heuristics to find optimal paths efficiently

Business Example: Warehouse Navigation
-----------------------------------
We'll demonstrate using a warehouse robot that needs to:
- Navigate through a warehouse layout
- Avoid obstacles (stored items)
- Find optimal paths to pick locations

This example shows how to:
1. Represent the warehouse as a grid
2. Implement breadth-first search for pathfinding
3. Visualize the search process and solution
""")

class BusinessSearch:
    """
    Demonstrates different search algorithms using a simplified 
    supply chain optimization problem.
    """
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        # Create a more interesting warehouse layout
        self.grid[1, 1:4] = 1  # Storage area 1
        self.grid[3, 2:4] = 1  # Storage area 2
        self.grid[2, 0] = 1    # Additional obstacle
    
    def breadth_first_search(self, start, goal):
        """
        Implement BFS to find path from start to goal in warehouse.
        Used for optimal path finding in warehouse navigation.
        """
        queue = deque([[start]])
        visited = set([start])
        
        while queue:
            path = queue.popleft()
            row, col = path[-1]
            
            if (row, col) == goal:
                return path
            
            # Check all adjacent cells
            for next_row, next_col in [
                (row+1, col), (row-1, col),
                (row, col+1), (row, col-1)
            ]:
                if (0 <= next_row < self.size and 
                    0 <= next_col < self.size and 
                    self.grid[next_row, next_col] != 1 and
                    (next_row, next_col) not in visited):
                    queue.append(path + [(next_row, next_col)])
                    visited.add((next_row, next_col))
        return None
    
    def visualize_search(self, path=None):
        """Visualize the warehouse layout and found path"""
        plt.figure(figsize=(10, 10))
        # Create a more informative visualization
        plt.imshow(self.grid, cmap='Greys')
        
        # Add grid lines
        for i in range(self.size):
            plt.axhline(i-0.5, color='black', linewidth=0.5)
            plt.axvline(i-0.5, color='black', linewidth=0.5)
        
        # Show path if found
        if path:
            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], 'r-', linewidth=3, label='Found Path')
            plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
            plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='Goal')
        
        plt.title('Warehouse Navigation Map')
        plt.legend()
        plt.grid(True)
        plt.show()

def reflection_search_algorithms():
    print("""
Reflection Questions - Search Algorithms:
--------------------------------------
1. How would the path change if we used a different search algorithm,
   such as depth-first search?
2. What business metrics could we use to evaluate different pathfinding
   algorithms beyond just path length?
3. How could this algorithm be adapted for multi-robot coordination in
   the warehouse?
""")

# Part 3: Local Search
# ===================

class ResourceOptimization:
    """
    Demonstrates local search algorithms using a resource allocation problem.
    Shows how to optimize business resource distribution.
    """
    def __init__(self, n_resources=5, n_departments=3):
        self.n_resources = n_resources
        self.n_departments = n_departments
        # Initial random allocation
        self.current_allocation = np.random.randint(0, n_resources, n_departments)
        while sum(self.current_allocation) > n_resources:
            self.current_allocation = np.random.randint(0, n_resources, n_departments)
    
    def objective_function(self, allocation):
        """
        Calculate the effectiveness of a resource allocation.
        Higher return = better allocation
        """
        if sum(allocation) > self.n_resources:
            return float('-inf')
        
        # Simplified business metric: balance between departments
        # while maximizing resource usage
        usage_score = sum(allocation) / self.n_resources
        balance_score = 1 / (np.std(allocation) + 1)
        return usage_score + balance_score
    
    def hill_climbing(self, max_iterations=100):
        """Implement hill climbing for resource optimization"""
        current = self.current_allocation.copy()
        current_score = self.objective_function(current)
        
        improvements = []
        for i in range(max_iterations):
            # Generate a neighbor by moving one resource
            neighbor = current.copy()
            # Randomly select source and destination departments
            src, dst = np.random.randint(0, self.n_departments, 2)
            if neighbor[src] > 0:
                neighbor[src] -= 1
                neighbor[dst] += 1
                
            neighbor_score = self.objective_function(neighbor)
            
            if neighbor_score > current_score:
                current = neighbor
                current_score = neighbor_score
                improvements.append(current_score)
            
        return current, improvements
    
    def visualize_optimization(self, improvements):
        """Visualize the optimization process"""
        plt.figure(figsize=(10, 5))
        plt.plot(improvements)
        plt.title('Resource Allocation Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Score')
        plt.show()

# Part 4: Adversarial Search
# =========================

class MarketCompetition:
    """
    Demonstrates adversarial search concepts using a simplified
    market competition scenario between two businesses.
    """
    def __init__(self, market_size=5):
        self.market_size = market_size
        self.state = np.zeros(market_size)  # Market share distribution
        
    def get_actions(self, player):
        """Get possible market actions for a player"""
        return ['invest', 'maintain', 'reduce']
    
    def minimax(self, state, depth, maximizing_player):
        """
        Implement minimax for market strategy optimization.
        Demonstrates how businesses can plan moves considering competitor responses.
        """
        if depth == 0:
            return self.evaluate_state(state)
        
        if maximizing_player:
            value = float('-inf')
            for action in self.get_actions(True):
                new_state = self.apply_action(state, action, True)
                value = max(value, self.minimax(new_state, depth-1, False))
            return value
        else:
            value = float('inf')
            for action in self.get_actions(False):
                new_state = self.apply_action(state, action, False)
                value = min(value, self.minimax(new_state, depth-1, True))
            return value
    
    def evaluate_state(self, state):
        """Evaluate market state - simplified scoring"""
        return np.sum(state)
    
    def apply_action(self, state, action, is_player_one):
        """Apply market action and return new state"""
        new_state = state.copy()
        if action == 'invest':
            idx = 0 if is_player_one else -1
            new_state[idx] += 1
        elif action == 'reduce':
            idx = 0 if is_player_one else -1
            new_state[idx] = max(0, new_state[idx] - 1)
        return new_state

def main():
    print_header()
    
    # Topic 1: Problem Formulation
    explain_problem_formulation()
    route_problem = BusinessRoute()
    route_problem.visualize_problem()
    reflection_problem_formulation()
    
    # Topic 2: Search Algorithms
    explain_search_algorithms()
    search_problem = BusinessSearch()
    path = search_problem.breadth_first_search((0,0), (4,4))
    search_problem.visualize_search(path)
    reflection_search_algorithms()
    
    # Demonstrate Local Search
    print("\n3. Local Search Example: Resource Optimization")
    print("=========================================")
    optimization_problem = ResourceOptimization()
    final_allocation, improvements = optimization_problem.hill_climbing()
    optimization_problem.visualize_optimization(improvements)
    
    # Demonstrate Adversarial Search
    print("\n4. Adversarial Search Example: Market Competition")
    print("===========================================")
    market_problem = MarketCompetition()
    best_value = market_problem.minimax(market_problem.state, depth=3, maximizing_player=True)
    print(f"Optimal market strategy value: {best_value}")
    
    print("\nCoding Challenge")
    print("===============")
    print("""
Final Business Challenge:
----------------------
Implement a warehouse optimization system that combines concepts from all topics:
1. Problem Formulation: Define the warehouse space and constraints
2. Search Algorithms: Find optimal paths for robots
3. Local Search: Optimize resource allocation
4. Adversarial Search: Handle multiple competing objectives

Your task is to implement a solution that balances:
- Efficient path finding
- Resource utilization
- Cost minimization
- Real-world constraints

Template provided in the accompanying worksheet.
""")

if __name__ == "__main__":
    main() 