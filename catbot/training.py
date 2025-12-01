import random
import time
from typing import Dict
import numpy as np
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################








#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1, *, alpha_param: float = None, gamma_param: float = None, min_epsilon_param: float = None, decay_rate_param: float = None, optimistic_init: float = None, terminal_reward_param: float = None):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    # Optionally use optimistic initialization to encourage exploration of actions
    init_val = 0.0 if optimistic_init is None else float(optimistic_init)
    q_table: Dict[int, np.ndarray] = {
        state: np.full(env.action_space.n, init_val, dtype=float) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    









    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        # Hyperparameters for Q-learning (allow overrides via function args)
        alpha = 0.5 if alpha_param is None else float(alpha_param)
        gamma = 0.95 if gamma_param is None else float(gamma_param)
        # epsilon-greedy params
        if ep == 1:
            epsilon = 1.0
            min_epsilon = 0.05 if min_epsilon_param is None else float(min_epsilon_param)
            decay_rate = 0.9992 if decay_rate_param is None else float(decay_rate_param)  # multiplicative decay per episode

        obs, _ = env.reset()

        max_steps = 60

        for step in range(max_steps):
            # choose action (epsilon-greedy)
            if random.random() < epsilon:
                action = random.randrange(env.action_space.n)
            else:
                action = int(np.argmax(q_table[obs]))

            next_obs, _, terminated, truncated, _ = env.step(action)

            # decode positions from obs and next_obs
            def decode(state_int):
                ag_r = (state_int // 1000) % 10
                ag_c = (state_int // 100) % 10
                cat_r = (state_int // 10) % 10
                cat_c = state_int % 10
                return ag_r, ag_c, cat_r, cat_c

            ag_r, ag_c, cat_r, cat_c = decode(obs)
            nag_r, nag_c, ncat_r, ncat_c = decode(next_obs)

            prev_dist = abs(ag_r - cat_r) + abs(ag_c - cat_c)
            next_dist = abs(nag_r - ncat_r) + abs(nag_c - ncat_c)

            # reward shaping: encourage shorter catches and moving closer
            reward = -1.0  # small per-step penalty to encourage speed
            if next_dist < prev_dist:
                reward += 2.0
            elif next_dist > prev_dist:
                reward -= 0.5

            if terminated or truncated:
                # if episode ended, give a strong positive reward for catching
                # (env sets done True only when agent catches cat)
                if np.array_equal((nag_r, nag_c), (ncat_r, ncat_c)):
                    tr = 100.0 if terminal_reward_param is None else float(terminal_reward_param)
                    reward += tr

            # Q-update
            old_value = q_table[obs][action]
            next_max = 0.0 if (terminated or truncated) else float(np.max(q_table[next_obs]))
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[obs][action] = new_value

            obs = next_obs

            if terminated or truncated:
                break

        # decay epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        # occasional logging to track progress (kept minimal)
        if ep % 500 == 0:
            print(f"episode {ep}/{episodes} - epsilon {epsilon:.3f}")
















        
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table