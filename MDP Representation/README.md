# MDP REPRESENTATION

## AIM:
To represent a Markov Decision Process(MDP) problem.

## PROBLEM STATEMENT:

### Problem Description
The goal of this problem is to develop a policy that efficiently manages the engine's heat level in a car while optimizing the expected cumulative reward.

### State Space
The state space consists of two possible states:
- "Cool" (representing a cool engine temperature)
- "Hot" (representing a hot engine temperature)

### Sample State
A sample state might be "Cool," indicating that the engine's temperature is currently at a safe and acceptable level.

### Action Space
The action space includes two available actions:
- "Accelerate" (to increase the engine's heat level)
- "Brake" (to decrease the engine's heat level)

### Sample Action
A sample action could be "Brake," indicating that the driver has chosen to apply the brakes in order to reduce the engine's heat level.

### Reward Function
The rewards are structured to encourage actions that maintain the engine in the "Cool" state (reward of 1.0) and penalize actions that lead to the "Hot" state (strong penalty of -10.0).

### Graphical Representation
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/c6ec3554-ea31-4e5a-b72c-fcf50fcb470e)

## PYTHON REPRESENTATION:
```python3
mdp = {
    "Cool": {
        "Accelerate": [
            (0.2, "Cool", -10.0, False),
            (0.8, "Hot", -10.0, False),
        ],
        "Brake": [
            (0.9, "Cool", 1.0, False),
            (0.1, "Hot", -10.0, False),
        ],
    },
    "Hot": {
        "Accelerate": [
            (0.6, "Cool", 1.0, False),
            (0.4, "Hot", -10.0, False),
        ],
        "Brake": [
            (0.3, "Cool", 1.0, False),
            (0.7, "Hot", -10.0, False),
        ],
    },
}
```

## OUTPUT:
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/624c8f2e-b866-4f5c-b4be-e756aa02e30e)

## RESULT:
Thus, a Markov Decision Process (MDP) problem is represented in the following ways:

- Text representation
- Graphical representation
- Python - Dictonary representation
