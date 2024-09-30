#! python3

import numpy as np
import matplotlib.pyplot as plt



# Use the run exploration algorithm we have provided to get you
# started, this returns the expected rewards after running an exploration 
# algorithm in the K-Armed Bandits problem. We have already specified a number
# of parameters specific to the 10-armed testbed for guidance.



def runExplorationAlgorithm(explorationAlgorithm, param, iters):
    cumulativeRewards = []
    for i in range(iters):
        # number of time steps
        t = 1000
        # number of arms, 10 in this instance
        k = 10
        # mean reward across each of the K arms
        # sample the actual rewards from a normal distribution with mean of meanRewards and standard deviation of 1
        meanRewards = np.random.normal(1,1,k)
        # counts for each arm
        n = np.zeros(k)
        # extract expected rewards by running specified exploration algorithm with the parameters above
        # param is the different, specific parameter for each exploration algorithm
        # this would be epsilon for epsilon greedy, initial values for optimistic intialization, c for UCB, and temperature for Boltmann
        currentRewards = explorationAlgorithm(param, t, k, meanRewards, n)
        cumulativeRewards.append(currentRewards)
    # calculate average rewards across each iteration to produce expected rewards
    expectedRewards = np.mean(cumulativeRewards, axis=0)
    return expectedRewards



def epsilonGreedyExploration(epsilon, steps, k, meanRewards, n):
    # TODO implement the epsilong greedy algorithm over all steps and return
    # the expected rewards across all steps
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION

    # q estimate for k=10 arms
    q_ests = np.zeros(k)

    # run epsilon greedy algorithm
    for i in range(steps):

        # get epsilon greedy full timestep return
        reward = 0
        reward += (1 - epsilon) * meanRewards[np.argmax(q_ests)]

        for arm in range(k):
            reward += (1 / k) * epsilon * meanRewards[arm]
        expectedRewards[i] = reward

        # action selection - best arm estimate from table
        if (np.random.rand() > epsilon):
            action = np.argmax(q_ests)
        # with epsilon prob force exploration
        else:
            action = np.random.randint(k)

        # update reward with noise
        true_reward = meanRewards[action] + np.random.normal()
        n[action] += 1
        q_ests[action] += (1 / n[action]) * (true_reward - q_ests[action])


    # END STUDENT SOLUTION
    return(expectedRewards)



def optimisticInitialization(value, steps, k, meanRewards, n):
    # TODO implement the optimistic initializaiton algorithm over all steps and
    # return the expected rewards across all steps
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION

    # q estimate for k=10 arms
    q_ests = np.zeros(k)

    # run algorithm
    for i in range(steps):

        # action selection - best arm estimate from table
        action = np.argmax(q_ests)
        expectedRewards[i] = meanRewards[action]

        # calculate reward
        reward = meanRewards[action] + np.random.normal()
        n[action] += 1
        q_ests[action] += (1 / n[action]) * (reward - q_ests[action])


    # END STUDENT SOLUTION
    return(expectedRewards)



def ucbExploration(c, steps, k, meanRewards, n):
    # TODO implement the UCB exploration algorithm over all steps and return the
    # expected rewards across all steps, remember to pull all arms initially
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION

    # q estimate for k=10 arms
    q_ests = np.zeros(k)
    n += 1

    # run algorithm
    for i in range(steps):
        # action selection - best arm estimate from table
        action = np.argmax(q_ests + c * np.sqrt(np.log(i + 1) / n))

        expectedRewards[i] = meanRewards[action]

        # calculate reward
        reward = meanRewards[action] + np.random.normal()
        n[action] += 1
        q_ests[action] += (1 / n[action]) * (reward - q_ests[action])

    # END STUDENT SOLUTION
    return(expectedRewards)



def boltzmannExploration(temperature, steps, k, meanRewards, n):
    # TODO implement the Boltzmann Exploration algorithm over all steps and
    # return the expected rewards across all steps
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION

    # q estimate for k=10 arms
    q_ests = np.zeros(k)
    k_probs = np.ones(k) / k


    # run algorithm
    for i in range(steps):
        for arm in range(k):
            k_probs[arm] = np.exp(temperature * q_ests[arm]) / np.exp(temperature * q_ests).sum()

        # action selection
        action = np.random.choice(k, p=k_probs)
        expectedRewards[i] = k_probs.dot(meanRewards)

        # calculate reward
        reward = meanRewards[action] + np.random.normal()
        n[action] += 1
        q_ests[action] += (1 / n[action]) * (reward - q_ests[action])


    # END STUDENT SOLUTION
    return(expectedRewards)



# plot template
def plotAlgorithms(alg_param_list, explorationAlgorithm):
    # TODO given a list of (algorithm, parameter) tuples, make a graph that
    # plots the expectedRewards of running that algorithm with those parameters
    # iters times using runExplorationAlgorithm plot all data on the same plot
    # include correct labels on your plot
    iters = 1000
    alg_to_name = {epsilonGreedyExploration : 'Epsilon Greedy Exploration',
                   optimisticInitialization : 'Optimistic Initialization',
                   ucbExploration: 'UCB Exploration',
                   boltzmannExploration: 'Boltzmann Exploration'}
    # BEGIN STUDENT SOLUTION

    X = np.arange(1, iters+1)
    # calculate your Ys (expected rewards) per each parameter value
    # plot all the Ys on the same plot
    # include correct labels on your plot!

    for param in alg_param_list:
        Y = runExplorationAlgorithm(explorationAlgorithm, param, iters=iters)
        plt.plot(X, Y, label=param)
        plt.ylabel('Expected rewards')
        plt.xlabel('Timesteps')

        # print( ''.join(alg_to_name[epsilonGreedyExploration].split()) )
        if (explorationAlgorithm == epsilonGreedyExploration):
            plt.legend(title='epsilon', loc=4)
            plt.title(alg_to_name[epsilonGreedyExploration])
            plt.savefig(f'{alg_to_name[epsilonGreedyExploration]}.png', dpi=300)
        elif (explorationAlgorithm == optimisticInitialization):
            plt.legend(title='optimistic')
            plt.title(alg_to_name[optimisticInitialization])
            plt.savefig(f'{alg_to_name[optimisticInitialization]}.png', dpi=300)
        elif (explorationAlgorithm == ucbExploration):
            plt.legend(title='c')
            plt.title(alg_to_name[ucbExploration])
            plt.savefig(f'{alg_to_name[ucbExploration]}.png', dpi=300)
        elif (explorationAlgorithm == boltzmannExploration):
            plt.legend(title='temp')
            plt.title(alg_to_name[boltzmannExploration])
            plt.savefig(f'{alg_to_name[boltzmannExploration]}.png', dpi=300)
    plt.show()
    plt.close()

    # END STUDENT SOLUTION
    pass



if __name__ == '__main__':
    # TODO call plotAlgorithms here to plot your algorithms
    np.random.seed(10003)
    iters = 1000

    # BEGIN STUDENT SOLUTION
    plotAlgorithms([0, 0.001, 0.01, 0.1, 1.0], epsilonGreedyExploration)
    plotAlgorithms([0, 1, 2, 5, 10], optimisticInitialization)
    plotAlgorithms([0, 1, 2, 5], ucbExploration)
    plotAlgorithms([1, 3, 10, 30, 100], boltzmannExploration)


    ##### Q2.5 comparison plot - select best param and compare all algorithms
    Y_ep = runExplorationAlgorithm(epsilonGreedyExploration, 0.01, iters=1000)
    Y_op = runExplorationAlgorithm(optimisticInitialization, 5, iters=1000)
    Y_ucb = runExplorationAlgorithm(ucbExploration, 1, iters=1000)
    Y_bol = runExplorationAlgorithm(boltzmannExploration, 3, iters=1000)


    X = np.arange(1, iters+1)

    plt.plot(X, Y_ep, label='epsilon greedy = 0.01')
    plt.plot(X, Y_op, label='optimistic = 5')
    plt.plot(X, Y_ucb, label='ucb = 1')
    plt.plot(X, Y_bol, label='boltzmann = 3')

    plt.legend()
    plt.ylabel('Expected rewards')
    plt.xlabel('Timesteps')
    plt.title('Algorithm comparison plot')
    plt.savefig(f'alg_comparison_plot.png', dpi=300)
    plt.show()
    plt.close()

    # END STUDENT SOLUTION
    pass
