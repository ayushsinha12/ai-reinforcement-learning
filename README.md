# Escape the Castle: MDP & Q-Learning Agent

A Python-based reinforcement learning project that reconstructs a 5×5 grid escape MDP via Monte Carlo sampling and uses tabular Q-learning to discover an optimal escape policy.

## Overview

In **Escape the Castle**, you navigate from the top-left corner (0, 0) to the bottom-right corner (4, 4) of a 5×5 grid populated with four hidden guards. You never know a guard’s strength or detection ability until you encounter them. Every move and interaction is governed by an underlying Markov Decision Process (MDP) with probabilistic outcomes. This repository contains:

- A model-based estimator that computes per-guard defeat probabilities via Monte Carlo sampling.  
- A tabular Q-learning agent that learns an optimal policy through trial-and-error.  
- A lightweight PyGame viewer to watch trained agents play the game (optional).

---

## Environment & Constraints

- **Grid world**: 25 cells [(0, 0) → (4, 4)], start at (0, 0), goal at (4, 4).  
- **Hidden guards**: Four guards are randomly placed (excluding start and goal). You only discover them by entering the same cell.  
- **Actions**: `UP`, `DOWN`, `LEFT`, `RIGHT` (90 % success, 10 % “slip” to a random adjacent cell), plus `HIDE` and `FIGHT` when sharing a cell with a guard.  
- **States**: Encoded by (x, y, health_state, guard_in_cell) → hashed into [0…374].  
- **API rules**: You may only call `env.reset()`, `env.step()`, `env.action_space.sample()` and use provided hash functions. Direct modification of environment internals is forbidden.

---

## Features

- **Model-Based Reconstruction**  
  Estimate each guard’s defeat probability under the `FIGHT` action by running random-policy episodes and aggregating outcomes.  

- **Tabular Q-Learning**  
  Implement an ε-greedy agent with per-state-action learning rates η = 1/(1 + visits) and decaying ε to converge on an optimal escape policy.  

- **Visualization**  
  Optionally launch a PyGame front-end to watch your agent navigate the castle in real time.

---

## Results

- **Guard defeat probabilities** (reconstruct_MDP):  
  - G1: 0.72  
  - G2: 0.65  
  - G3: 0.80  
  - G4: 0.58  

- **Average reward** over 100 000 Q-learning episodes: 861.1 (expected ≥ 860.0)  
- **Autograder scores**: 16/16 for MDP reconstruction, 24/24 for Q-learning

## File Structure

```bash
ai-reinforcement-learning/
├── code/  
│   ├── mdp_gym.py            # Provided MDP environment and hashing utilities  
│   ├── reconstruct_MDP.py    # Monte Carlo estimator for guard‐defeat probabilities  
│   ├── Q_learning.py         # Tabular Q-learning training script  
│   ├── vis_gym.py            # PyGame viewer for a trained policy (optional)  
│   ├── Q_table.pickle        # Saved Q-table (output of Q_learning.py)  
│   └── __pycache__/          # Compiled bytecode (you can ignore or delete this)  
└── README.md                 # Project documentation (this file)
