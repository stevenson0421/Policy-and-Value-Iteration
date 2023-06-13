import numpy as np
import gymnasium
from tqdm import trange
import cv2
import os

def get_rewards_and_transitions(environment):
    # get state and action size
    number_of_state = environment.observation_space.n
    number_of_action = environment.action_space.n

    # Initialize reward and transition function
    rewards = np.zeros((number_of_state, number_of_action))
    transitions = np.zeros((number_of_state, number_of_action, number_of_state))

    # get rewards and transitions
    for state in range(number_of_state):
        for action in range(number_of_action):
            for transition_info in environment.P[state][action]:
                transition, next_state, reward, done = transition_info
                rewards[state, action] = reward
                transitions[state, action, next_state] = transition

    return rewards, transitions

def policy_evaluation(number_of_state,
                      number_of_action,
                      values,
                      rewards,
                      transitions,
                      discount_factor=0.99,
                      max_iteration=10**6,
                      tolerance=10**-3):

    # value iteration
    for i in trange(max_iteration):
        values_old = values.copy()

        values = np.max(rewards + np.matmul(transitions, discount_factor * values_old), axis=1)

        if np.max(np.abs(values - values_old)) < tolerance:
            break
    
    return values

def test_policies(environment, policies, test_iterations=1, record_path='./', record_name='video', record_frame=2):
    trajectory_reward = 0
    trajectory_length = 0
    
    output_screens = []

    max_trajectory_reward = -10**8

    for i in trange(test_iterations):
        screens = []
        state, info = environment.reset(seed=i)
        while True:
            action = policies[state]
            next_state, reward, terminated, truncated, info = environment.step(action)

            trajectory_reward += reward
            trajectory_length += 1

            screens.append(environment.render())

            state = next_state

            if (terminated or truncated):
                print(f'trajectory ends with length {trajectory_length}: reward: {trajectory_reward}')
                if trajectory_reward > max_trajectory_reward:
                    output_screens = screens
                if i == test_iterations-1:
                    if not os.path.exists(record_path):
                        os.makedirs(record_path)
                    out = cv2.VideoWriter(os.path.join(record_path, f'{record_name}.avi'),cv2.VideoWriter_fourcc(*'DIVX'), record_frame, (screens[0].shape[1], screens[0].shape[0]))
                    for img in output_screens:
                        out.write(img)
                    out.release()
                
                trajectory_reward = 0
                trajectory_length = 0

                break

def value_iteration(environment=gymnasium.make('Taxi-v3', render_mode='rgb_array'),
                    discount_factor=0.99,
                    max_iteration=10**6,
                    tolerance=10**-3,
                    probabilistic=False):
    
    number_of_state = environment.observation_space.n
    number_of_action = environment.action_space.n
    
    # Initialize with a random policy
    policies = np.array([environment.action_space.sample() for state in range(number_of_state)])

    # Initialize value was zero
    values = np.zeros(number_of_state)

    rewards, transitions = get_rewards_and_transitions(environment=environment)

    # value iteration (policy evaluation)
    values = policy_evaluation(number_of_state=number_of_state,
                               number_of_action=number_of_action,
                               values=values,
                               rewards=rewards,
                               transitions=transitions,
                               discount_factor=discount_factor,
                               max_iteration=max_iteration,
                               tolerance=tolerance)

    policies = np.argmax(rewards + np.matmul(transitions, discount_factor * values), axis=1)

    return policies

def policy_iteration(environment=gymnasium.make('Taxi-v3', render_mode='rgb_array'),
                    discount_factor=0.99,
                    max_iteration=10**6,
                    tolerance=10**-3,
                    probabilistic=False):
    
    number_of_state = environment.observation_space.n
    number_of_action = environment.action_space.n
    
    # Initialize with a random policy
    policies = np.array([environment.action_space.sample() for state in range(number_of_state)])

    # Initialize value was zero
    values = np.zeros(number_of_state)

    rewards, transitions = get_rewards_and_transitions(environment=environment)

    # policy iteration
    for i in range(max_iteration):
        # policy evaluation
        values = policy_evaluation(number_of_state=number_of_state,
                                   number_of_action=number_of_action,
                                   values=values,
                                   rewards=rewards,
                                   transitions=transitions,
                                   discount_factor=discount_factor,
                                   max_iteration=max_iteration,
                                   tolerance=tolerance)

        # policy improvement
        policies_old = policies.copy()
        
        policies = np.argmax(rewards + np.matmul(transitions, discount_factor * values), axis=1)

        
        # stop iteration if policy is stable
        if np.all(np.equal(policies, policies_old)):
            break
        

    return policies

if __name__ == '__main__':
    environment = gymnasium.make('FrozenLake-v1', render_mode='rgb_array', is_slippery=False)
    environment_name = environment.unwrapped.spec.id.replace('/', '_')
    record_path=f'./video/{environment_name}'
    record_frame=2

    vi_policies = value_iteration(environment=environment)
    print('Value Iteration Completed')
    test_policies(environment=environment, policies=vi_policies, record_path=record_path, record_name='value_iteration', record_frame=record_frame)
    pi_policies = policy_iteration(environment=environment)
    print('Policy Iteration Completed')
    test_policies(environment=environment, policies=pi_policies, record_path=record_path, record_name='policy_iteration', record_frame=record_frame)
    print(f'discrepancy: {np.sum(np.abs(vi_policies - pi_policies))}')