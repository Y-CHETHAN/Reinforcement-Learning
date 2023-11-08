# MONTE CARLO CONTROL ALGORITHM

## AIM:
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.

## PROBLEM STATEMENT:
To develop a Python program that uses the Monte Carlo algorithm to find the optimal policy for a given Reinforcement Learning environment. The environment consists of a grid layout with 5 terminal states, including a goal state and a hazardous state that the agent must avoid. The agent can take 4 possible actions, and the transition probabilities are such that there is a 33.3% chance of moving as intended and a 66.6% chance of moving in orthogonal directions. The agent receives a reward of 1 for reaching the goal state and a reward of 0 otherwise. The Monte Carlo Control Algorithm is used to find the optimal policy, and the function mc_control is provided to implement the algorithm. The function takes several parameters, including the environment, gamma, alpha, and epsilon values, and the number of episodes and maximum steps. The function returns the Q-table and a tracking array.



## MONTE CARLO CONTROL ALGORITHM:
1. Initialize the Q-table with arbitrary values for each state-action pair.
2. For each episode:
   * Initialize the episode with the starting state.
   * Generate an episode by following the current policy until a terminal state is reached.
   * For each state-action pair in the episode:
     * Calculate the return (total reward from that state-action pair until the end of the episode).
     * Update the Q-value for that state-action pair using the return and the current Q-value.
    * Update the policy based on the updated Q-table.
4. Repeat step 2 for a specified number of episodes.
5. Return the Q-table, which represents the optimal policy.

## MONTE CARLO CONTROL FUNCTION:
```python
import numpy as np
from tqdm import tqdm

def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    # Get the number of states and actions
    nS, nA = env.observation_space.n, env.action_space.n

    # Create an array for discounting
    disc = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    def decay_schedule(init_value, min_value, decay_ratio, n):
        return np.maximum(min_value, init_value * (decay_ratio ** np.arange(n))

    # Create schedules for alpha and epsilon decay
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    # Initialize Q-table and tracking array
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    def select_action(state, Q, epsilon):
        return np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(nA)

    for e in tqdm(range(n_episodes), leave=False):
        # Generate a trajectory
        traj = gen_traj(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=np.bool)

        for t, (state, action, reward, _, _) in enumerate(traj):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            n_steps = len(traj[t:])
            G = np.sum(disc[:n_steps] * traj[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        Q_track[e] = Q

    # Calculate the value function and policy
    V = np.max(Q, axis=1)
    pi = {s: np.argmax(Q[s]) for s in range(nS)}

    return Q, V, pi
```

## OUTPUT:
### Optimal Policy
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/3e962669-43da-46de-be5a-a2ae2518f80b)

### Optimal Value Function
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/4543693a-8bee-49b6-af7b-d78977263b9a)

### Success Rate for the Optimal Policy
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/c1ffd1a6-0aca-48a8-b2b3-a60897694b6c)

## RESULT:
Thus, a python program is developed to find the optimal policy for the given RL environment using the Monte Carlo Control Algorithm.
