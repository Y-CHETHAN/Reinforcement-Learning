# SARSA Learning Algorithm

## AIM:
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT:
Consider the gridworld environment shown below. The agent starts in the starting state (S) and must navigate to the goal state (G) while avoiding the hole state (H). The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states. The agent can take two actions: move right (R) or move left (L). The transition probabilities for each action are as follows: 50% chance that the agent moves in the intended direction, 33.33% chance that the agent stays in its current state, and 16.66% chance that the agent moves in the opposite direction. For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2. Develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

![image](https://github.com/Marinto-Richee/Reinforcement-Learning/assets/65499285/d2857587-d2b8-4056-bd48-c51a0013a7c1)


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

### State-value function found by Sarsa

### FVMC estimates through time vs. true values
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/38de5b20-1eae-4d9c-8092-3fab9ed4b1c3)

### Sarsa estimates through time vs. true values
![image](https://github.com/Y-CHETHAN/Reinforcement-Learning/assets/75234991/5b800e5f-c2c5-4a03-8aae-af74fc00fa4f)

## RESULT:
Thus, the optimal policy for the given RL environment is found using SARSA-Learning and the state values are compared with the Monte Carlo method.
