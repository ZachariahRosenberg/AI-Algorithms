import gym
import numpy as np

def policy_evaluation(policy, env, discount=1., theta=1e-9, max_iters=1e9):
    print(policy)
    print(env.unwrapped.P[4])
    # initialize counter
    eval_iters = 1

    # initialize state values
    V = np.zeros(env.unwrapped.nS)

    for i in range(int(max_iters)):
        delta = 0

        for state in range(env.unwrapped.nS):
            #initialize value
            v=0

            # iterate through every state, action
            for action, action_prob in enumerate(policy[state]):
                # look ahead to nex states value
                for state_prob, next_state, r, d in env.unwrapped.P[state][action]:
                    # add expected return
                    v += action_prob * state_prob * (r + discount * V[next_state])

            delta = max(delta, np.abs(V[state] - v))

            V[state] = v

        eval_iters += 1

        if delta < theta:
            #print(f'Policy evaluated in {eval_iters} iterations.')
            break

    return V

def policy_iteration(env, discount=1., max_iters=1e9):
    # initialize policy randomly
    policy = np.ones([env.unwrapped.nS, env.unwrapped.nA]) / env.unwrapped.nA

    #Evaluate policy
    evaluated_policies = 1
    for i in range(int(max_iters)):
        stable_policy=False

        V = policy_evaluation(policy, env, discount=discount, max_iters=max_iters)

        for state in range(len(V)):
            current_action = np.argmax(policy[state])
            action_value = one_step_lookahead(env, state, V, discount)
            best_action = np.argmax(action_value)
            if current_action != best_action:
                stable_policy=True
                policy[state] = np.eye(env.unwrapped.nA)[best_action]
        evaluated_policies +=1

        if stable_policy:
            #print(f'Evaluated {evaluated_policies} policies.')
            return policy, V

def value_iteration(env, discount=1., theta=1e-9, max_iters=1e9):
    V = np.zeros(env.unwrapped.nS)

    for i in range(int(max_iters)):
        delta = 0
        for state in range(len(V)):
            action_value = one_step_lookahead(env, state, V, discount)
            best_action_value = np.max(action_value)
            delta = max(delta, np.abs(V[state] - best_action_value))
            V[state] = best_action_value
        if delta < theta:
            print(f'Value-iteration converged at iteration#{i}.')
            break
    
    policy = np.zeros([env.unwrapped.nS, env.unwrapped.nA])
    for state in range(env.unwrapped.nS):
        action_value = one_step_lookahead(env, state, V, discount)
        best_action = np.argmax(action_value)

        policy[state, best_action] = 1.0
    
    return policy, V

def one_step_lookahead(env, state, V, discount):
    action_values = np.zeros(env.unwrapped.nA)

    for action in range(len(action_values)):
        for p, next_state, r, d in env.unwrapped.P[state][action]:
            action_values[action] += p * (r + discount * V[next_state])
    
    return action_values

def play_episodes(env, n_eps, policy):
    wins = 0
    total_r = 0
    for e in range(n_eps):
        d = False
        state = env.reset()
        while not d:
            action = np.argmax(policy[state])
            next_state, r, d, _ = env.step(action)
            total_r += r
            state = next_state
            if d and r == 1:
                wins+=1
    avg_r = total_r / n_eps

    return wins, total_r, avg_r


if __name__ == '__main__':
    n_eps = 1
    solvers = [('Policy Iteration', policy_iteration),
            ('Value Iteration', value_iteration)]

    for iter_name, iter_f, in solvers:
        env = gym.make('FrozenLake-v0')

        # set up Policy and Value of States
        policy, V = iter_f(env, max_iters=2)

        wins, total_r, avg_r = play_episodes(env, n_eps, policy)
        # print(f'{iter_name} :: number of wins over {n_eps} episodes = {wins}')
        # print(f'{iter_name} :: average reward over {n_eps} episodes = {avg_r} \n\n')