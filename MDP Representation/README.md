# MDP REPRESENTATION

## AIM:
Write your aim here

## PROBLEM STATEMENT:

### Problem Description
The problem statement for the stock trading agent is to buy and sell stocks to make a profit while considering transaction costs and market dynamics. The goal of the agent is to maximize the expected amount of profit over time.

### State Space
The state space of the MDP for the stock trading agent is the set of all possible stock prices within a given range, discretized into individual states. For example, the state space could be defined as prices between ₹100 and ₹200, with a granularity of ₹1 increment.

### Sample State
A sample state could be the price of a stock being ₹150.

### Action Space
The action space of the MDP for the stock trading agent is the set of all possible orders that the agent can place, taking into account the amount of money available for trading and any trading constraints. The actions can be defined as:

- Buy: Purchase a specified quantity of shares.
- Sell: Sell a specified quantity of shares.
- Hold: Take no action.

### Sample Action
A sample action could be the agent buying 100 shares of a stock, provided it has sufficient funds and adheres to any minimum buy quantities.

### Reward Function
The reward function calculates rewards based on the following criteria:

- Positive reward for an increase in the portfolio's profit.
- Negative reward for a decrease in the portfolio's profit.

### Graphical Representation
Write your answer here

## PYTHON REPRESENTATION:
```python3
stock_trading_mdp = {
    "LowPrice-LowPortfolio": {
        "Buy": [(0.7, "LowPrice-HighPortfolio", 10.0, False), (0.3, "LowPrice-LowPortfolio", 0.0, False)],
        "Sell": [(0.3, "LowPrice-HighPortfolio", 5.0, False), (0.7, "LowPrice-LowPortfolio", -2.0, False)],
        "Hold": [(1.0, "LowPrice-LowPortfolio", 0.0, False)],
    },
    "LowPrice-HighPortfolio": {
        "Buy": [(0.6, "LowPrice-HighPortfolio", 8.0, False), (0.4, "LowPrice-LowPortfolio", 3.0, False)],
        "Sell": [(0.2, "LowPrice-HighPortfolio", 4.0, False), (0.8, "LowPrice-LowPortfolio", -1.0, False)],
        "Hold": [(1.0, "LowPrice-HighPortfolio", 0.0, False)],
    },
    "HighPrice-LowPortfolio": {
        "Buy": [(0.3, "HighPrice-HighPortfolio", 12.0, False), (0.7, "HighPrice-LowPortfolio", -5.0, False)],
        "Sell": [(0.8, "HighPrice-HighPortfolio", 6.0, False), (0.2, "HighPrice-LowPortfolio", -3.0, False)],
        "Hold": [(1.0, "HighPrice-LowPortfolio", 0.0, False)],
    },
    "HighPrice-HighPortfolio": {
        "Buy": [(0.2, "HighPrice-HighPortfolio", 10.0, False), (0.8, "HighPrice-LowPortfolio", 4.0, False)],
        "Sell": [(0.6, "HighPrice-HighPortfolio", 8.0, False), (0.4, "HighPrice-LowPortfolio", 1.0, False)],
        "Hold": [(1.0, "HighPrice-HighPortfolio", 0.0, False)],
    },
}
```
In this representation:

- States are defined based on combinations of "Price" (Low or High) and "Portfolio" (Low or High).
- Actions include "Buy", "Sell", and "Hold".
- For each action in each state, there are lists of tuples representing transition probabilities, rewards, and whether the episode terminates (True or False).

## OUTPUT:
Write your Python output here

## RESULT:
Write your output here
