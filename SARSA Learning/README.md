# SARSA LEARNING ALGORITHM

## AIM:
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT:
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state 1. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes 1. The agent can take one of three actions in each state: move left, move right, or stay in place 1. The agent receives a reward of +1 for reaching the goal state and a reward of 0 for all other states 1. The goal of the agent is to learn a policy that maximizes the expected cumulative reward over time.

## SARSA LEARNING ALGORITHM:
1. Initialize the Q-table with zeros.
2. Start exploring actions: For each state, select any one among all possible actions for the current state (S).
3. Travel to the next state (S') as a result of that action (a).
4. For all possible actions from the state (S') select the one with the highest Q-value.
5. Update Q-table values using the equation.
6. Set the next state as the current state.
7. If goal state is reached, then end and repeat the process.


## SARSA LEARNING FUNCTION:
```python
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state,Q,epsilon: 
    			np.argmax(Q[state]) 
    			if np.random.random() > epsilon 
                else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)

    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)

    for e in tqdm(range(n_episodes),leave=False):
        state, done = env.reset(), False
        action = select_action(state,Q,epsilons[e])

        while not done:
            next_state,reward,done,_ = env.step(action)
            next_action = select_action(next_state,Q,epsilons[e])

            td_target = reward+gamma*Q[next_state][next_action]*(not done)

            td_error = td_target - Q[state][action]

            Q[state][action] = Q[state][action] + alphas[e] * td_error

            state, action = next_state,next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q,axis=1))

    V = np.max(Q,axis=1)
    pi = lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
### State-value function found by FVMC
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/6f4e0111-4f8b-41c2-8606-52546dd14eb1)

### State-value function found by Sarsa
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/809889df-a246-4108-a56b-7c7929b5c9f3)

### FVMC estimates through time vs. true values
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/38de5b20-1eae-4d9c-8092-3fab9ed4b1c3)

### Sarsa estimates through time vs. true values
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/5b800e5f-c2c5-4a03-8aae-af74fc00fa4f)

## RESULT:
Thus, the optimal policy for the given RL environment is found using SARSA-Learning and the state values are compared with the Monte Carlo method.

