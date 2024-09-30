#! python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium

import lake_info



def value_func_to_policy(env, gamma, value_func):
    '''
    Outputs a policy given a value function.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute the policy for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.

    Returns
    -------
    np.ndarray
        An array of integers. Each integer is the optimal action to take in
        that state according to the environment dynamics and the given value
        function.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION
    return(policy)


# Q1.2 - policy iteration
def evaluate_policy_sync(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''

    # BEGIN STUDENT SOLUTION
    steps = 0
    delta = 1
    # set alg stopping condition
    while delta >= tol and steps < max_iters:
        val_func_cp = value_func
        # load env map, loop to update all states each sweep
        for state in range(env.observation_space.n):
            state_val = value_func[state]
            action = policy[state]
            val_func_cp[state] = 0
            # calc value changes
            for n_mat in env.P[state][action]:
                prob, n_state, reward, _ = n_mat
                val_func_cp[state] += prob * (reward + gamma * value_func[n_state])
                delta = max(delta, abs(state_val - val_func_cp[state]))
        # update new values for all states
        value_func = val_func_cp
        steps += 1
    # END STUDENT SOLUTION

    return(val_func_cp, steps)


# Q1.4 - policy iteration
def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''

    # BEGIN STUDENT SOLUTION
    val_func_cp = value_func
    steps = 0
    delta = 1
    # set alg stopping condition
    while delta >= tol and steps < max_iters:
        # load env map, loop to update all states each sweep
        for state in range(env.observation_space.n):
            state_val = value_func[state]
            action = policy[state]
            val_func_cp[state] = 0
            # calc value changes
            for n_mat in env.P[state][action]:
                prob, n_state, reward, _ = n_mat
                val_func_cp[state] += prob * (reward + gamma * value_func[n_state])
                delta = max(delta, abs(state_val - val_func_cp[state]))
        # update new values for all states
        value_func = val_func_cp
        steps += 1
    # END STUDENT SOLUTION

    return (val_func_cp, steps)


# Q1.4 - policy iteration
def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a policy. Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''

    # BEGIN STUDENT SOLUTION
    val_func_cp = value_func
    steps = 0
    delta = 1
    # set alg stopping condition
    while delta >= tol and steps < max_iters:
        # get randperm states
        perm_states = np.random.choice(env.observation_space.n, size=env.observation_space.n, replace=True)

        # load env map, loop to update all states each sweep
        for state in perm_states:
            state_val = value_func[state]
            action = policy[state]
            val_func_cp[state] = 0
            # calc value changes
            for n_mat in env.P[state][action]:
                prob, n_state, reward, _ = n_mat
                val_func_cp[state] += prob * (reward + gamma * value_func[n_state])
                delta = max(delta, abs(state_val - val_func_cp[state]))
        # update new values for all states
        value_func = val_func_cp
        steps += 1
    # END STUDENT SOLUTION

    return (val_func_cp, steps)


# Q1.2 - policy iteration
def improve_policy(env, gamma, value_func, policy):
    '''
    Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.
    policy: np.ndarray
        The policy to improve, maps states to actions.

    Returns
    -------
    (np.ndarray, bool)
        Returns the new policy and whether the policy changed.
    '''
    policy_changed = False

    # BEGIN STUDENT SOLUTION
    break_param = True
    policy_cp = policy

    # do sweep and update all states
    for state in range(env.observation_space.n):
        old_action = policy[state]
        max_r = -1
        next_action = -1
        # get action reward
        for action in env.P[state]:
            action_reward = 0
            for n_mat in env.P[state][action]:
                (prob, next_state, reward, _) = n_mat
                action_reward += prob * (reward + gamma * value_func[next_state])
            max_r = max(max_r, action_reward)
            # check param if deterministic greedy
            if max_r == action_reward:
                next_action = action
        policy_cp[state] = next_action
        # alg stopping condition
        if old_action != next_action:
            break_param = False
    # END STUDENT SOLUTION

    return(policy_cp, break_param, policy_changed)


# Q1.2 - policy iteration
def policy_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0

    # BEGIN STUDENT SOLUTION
    break_param = False
    # run pol it alg, check current pol against old, evaluate and improve
    while not break_param:
        value_func, steps = evaluate_policy_sync(env, value_func, gamma, policy)
        policy, break_param, _ = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        pe_steps += steps
    # END STUDENT SOLUTION

    return(policy, value_func, pi_steps, pe_steps)


# Q1.4 - policy iteration
def policy_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0

    # BEGIN STUDENT SOLUTION
    break_param = False
    # run pol it alg, check current pol against old, evaluate and improve
    while not break_param:
        value_func, steps = evaluate_policy_async_ordered(env, value_func, gamma, policy)
        policy, break_param, _ = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        pe_steps += steps
    # END STUDENT SOLUTION

    return (policy, value_func, pi_steps, pe_steps)


# Q1.4 - policy iteration
def policy_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0

    # BEGIN STUDENT SOLUTION
    break_param = False
    # run pol it alg, check current pol against old, evaluate and improve
    while not break_param:
        value_func, steps = evaluate_policy_async_randperm(env, value_func, gamma, policy)
        policy, break_param, _ = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        pe_steps += steps
    # END STUDENT SOLUTION

    return (policy, value_func, pi_steps, pe_steps)


# Q1.3 - value iteration
def value_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)

    # BEGIN STUDENT SOLUTION
    policy = np.zeros(env.env.observation_space.n)
    delta = 1
    steps = 0
    # loop for val iteration alg
    while delta >= tol and steps < max_iters:
        delta = 0
        steps += 1
        value_func_cp = value_func
        # get states for map
        for state in range(env.observation_space.n):
            val = value_func[state]
            max_val = 0
            max_ac = -1
            # get action value
            for action in env.P[state]:
                action_reward = 0
                for n_mat in env.P[state][action]:
                    prob, nstate, reward, _ = n_mat
                    action_reward += prob * (reward + gamma * value_func[nstate])
                max_val = max(max_val, action_reward)
                if max_val == action_reward:
                    max_ac = action
            value_func_cp[state] = max_val
            policy[state] = max_ac
            delta = max(delta, abs(val - value_func_cp[state]))
        value_func = value_func_cp
    # END STUDENT SOLUTION

    return(policy, value_func_cp, steps)


# Q1.5 - value iteration
def value_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)

    # BEGIN STUDENT SOLUTION
    policy = np.zeros(env.env.observation_space.n)
    delta = 1
    steps = 0
    # loop for val iteration alg
    while delta >= tol and steps < max_iters:
        delta = 0
        steps += 1
        # get states for map
        for state in range(env.observation_space.n):
            val = value_func[state]
            max_val = 0
            max_ac = -1
            # get action value
            for action in env.P[state]:
                action_reward = 0
                for n_mat in env.P[state][action]:
                    prob, nstate, reward, _ = n_mat
                    action_reward += prob * (reward + gamma * value_func[nstate])
                max_val = max(max_val, action_reward)
                if max_val == action_reward:
                    max_ac = action
            value_func[state] = max_val
            policy[state] = max_ac
            delta = max(delta, abs(val - value_func[state]))
    # END STUDENT SOLUTION

    return (policy, value_func, steps)


# Q1.5 - value iteration
def value_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)

    # BEGIN STUDENT SOLUTION
    policy = np.zeros(env.env.observation_space.n)
    delta = 1
    steps = 0
    # loop for val iteration alg
    while delta >= tol and steps < max_iters:
        delta = 0
        steps += 1

        # get randperm states
        perm_states = np.random.choice(env.observation_space.n, size=env.observation_space.n, replace=True)

        # get states for map
        for state in perm_states:
            val = value_func[state]
            max_val = 0
            max_ac = -1
            # get action value
            for action in env.P[state]:
                action_reward = 0
                for n_mat in env.P[state][action]:
                    prob, nstate, reward, _ = n_mat
                    action_reward += prob * (reward + gamma * value_func[nstate])
                max_val = max(max_val, action_reward)
                if max_val == action_reward:
                    max_ac = action
            value_func[state] = max_val
            policy[state] = max_ac
            delta = max(delta, abs(val - value_func[state]))
    # END STUDENT SOLUTION

    return (policy, value_func, steps)


# Q1.5 - value iteration
def value_iteration_async_custom(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION
    return(value_func, i)



# Here we provide some helper functions for your convinience.

def display_policy_letters(env, policy):
    '''
    Displays a policy as an array of letters.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    policy: np.ndarray
        The policy to display, maps states to actions.
    '''
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_info.actions_to_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.unwrapped.nrow, env.unwrapped.ncol)

    for row in range(env.unwrapped.nrow):
        print(''.join(policy_letters[row, :]))



def value_func_heatmap(env, value_func):
    '''
    Visualize a policy as a heatmap.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    value_func: np.ndarray
        The current value function estimate.
    '''
    fig, ax = plt.subplots(figsize=(7,6))

    # Reshape value_func to match the environment dimensions
    heatmap_data = np.reshape(value_func, [env.unwrapped.nrow, env.unwrapped.ncol])

    # Create a heatmap using Matplotlib
    cax = ax.matshow(heatmap_data, cmap='GnBu_r')

    # Set ticks and labels
    ax.set_yticks(np.arange(0, env.unwrapped.nrow))
    ax.set_xticks(np.arange(0, env.unwrapped.ncol))
    ax.set_yticklabels(np.arange(1, env.unwrapped.nrow + 1)[::-1])
    ax.set_xticklabels(np.arange(1, env.unwrapped.ncol + 1))

    # Display the colorbar
    cbar = plt.colorbar(cax)

    plt.show()



import seaborn as sns
if __name__ == '__main__':

    np.random.seed(10003)
    maps = lake_info.maps
    gamma = 0.9


    for map_name, map in maps.items():
        env = gymnasium.make('FrozenLake-v1', desc=map, map_name=map_name, is_slippery=False)

        n_grid = int(map_name.split('x')[0])

        # BEGIN STUDENT SOLUTION
        # Q1.2 sync policy iteration
        policy, val_func, imp_steps, val_steps = policy_iteration_sync(env, gamma)
        print(f'\nSync policy iteration {map_name}')
        print(f'Improvement epochs = {imp_steps}')
        print(f'Evaluation epochs = {val_steps}')
        display_policy_letters(env, policy)
        print(f'Value function')
        print(np.reshape(val_func, (n_grid, -1)), '\n')

        """
        # plot value function grid
        sns.heatmap(np.reshape(val_func, (n_grid, -1)), cmap=sns.cubehelix_palette(as_cmap=True), annot=True)
        plt.xticks(np.arange(1, n_grid+1), fontsize=10, labels=np.arange(1, n_grid+1))
        plt.yticks(np.arange(1, n_grid+1), fontsize=10, labels=np.arange(1, n_grid+1))
        plt.title('Sync policy iteration {map_name}\')
        plt.savefig(f'sync_pol_it_mat{n_grid}.png', dpi=300)
        plt.show()
        """


        # Q1.3 sync value iteration
        policy, val_func, steps = value_iteration_sync(env, gamma)
        print(f'\nSync value iteration {map_name}')
        print(f'Epochs = {steps}')
        display_policy_letters(env, policy)
        print(f'Value function')
        print(np.reshape(val_func, (n_grid, -1)), '\n')

        """
        # plot value function grid
        sns.heatmap(np.reshape(val_func, (n_grid, -1)), cmap=sns.cubehelix_palette(as_cmap=True), annot=True)
        plt.xticks(np.arange(1, n_grid+1), fontsize=10, labels=np.arange(1, n_grid+1))
        plt.yticks(np.arange(1, n_grid+1), fontsize=10, labels=np.arange(1, n_grid+1))
        plt.title('Sync value iteration {map_name}\')
        plt.savefig(f'sync_val_it_mat{n_grid}.png', dpi=300)
        plt.show()
        """


        # Q1.4 async policy iteration - ordered
        policy, val_func, imp_steps, val_steps = policy_iteration_async_ordered(env, gamma)
        print(f'\nAsync policy iteration - ordered {map_name}')
        print(f'Improvement epochs = {imp_steps}')
        print(f'Evaluation epochs = {val_steps}')
        display_policy_letters(env, policy)
        print(f'Value function')
        print(np.reshape(val_func, (n_grid, -1)), '\n')

        # Q1.4 async policy iteration - randperm
        randperm_imp_lst = []
        randperm_eval_lst = []

        for i in range(10):
            policy, val_func, imp_steps, val_steps = policy_iteration_async_randperm(env, gamma)
            randperm_imp_lst.append(imp_steps)
            randperm_eval_lst.append(val_steps)

        print(f'\nAsync policy iteration - randperm 10 avg {map_name}')
        print(f'Improvement epochs = {np.mean(np.array(randperm_imp_lst))}')
        print(f'Evaluation epochs = {np.mean(np.array(randperm_eval_lst))}\n')


        # Q1.5 async value iteration - ordered
        policy, val_func, steps = value_iteration_async_ordered(env, gamma)
        print(f'\nAsync value iteration - ordered {map_name}')
        print(f'Epochs = {steps}\n')

        # Q1.5 async value iteration - randperm
        randperm_val_step_lst = []

        for i in range(10):
            policy, val_func, steps = value_iteration_async_randperm(env, gamma)
            randperm_val_step_lst.append(steps)

        print(f'\nAsync value iteration - randperm {map_name}')
        print(f'Epochs = {steps}\n')

        # END STUDENT SOLUTION

