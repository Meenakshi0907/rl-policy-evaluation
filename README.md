# POLICY EVALUATION
## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.
### States
The environment has 7 states:

- Two Terminal States:
  * **G**: The goal state
  * **H**: A hole state.
- Five Transition states / Non-terminal States including
  * **S**: The starting state.
### Actions
The agent can take two actions:

- R: Move right.
- L: Move left.
### Transition Probabilities
The transition probabilities for each action are as follows:

* 50% chance that the agent moves in the intended direction.
* 33.33% chance that the agent stays in its current state.
* 16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/496de482-21df-41de-a186-b434cc84fb3f)

## POLICY EVALUATION FUNCTION
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/16bc8753-0c08-48be-8207-fc12747599a6)

## PROGRAM:
```py
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob,next_state,reward,done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
    return V
```
## OUTPUT:
### Policy 1 [pi_1]:
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/f7dac1dc-01e1-4fac-a231-2706cc5a2e50)
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/3f573dab-3abb-4d8c-b6c8-8867721f21e4)
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/b844bb1e-61a1-43f0-9a98-c3ca85c3f60e)

### Policy 2 [pi_2]:
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/e089c233-6ac2-4b5d-a7f0-093a96672d94)
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/a4a94f49-ff14-4b85-b44a-6334f44cc90a)
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/e6cf67c1-95fd-4516-8da9-399ac210c5b1)

### Conclusion:
![image](https://github.com/Meenakshi0907/rl-policy-evaluation/assets/94165108/85549ea3-d566-43f2-b17c-bc9b28eb60d0)

## RESULT:
Thus, a Python program is developed to evaluate the given policy.
