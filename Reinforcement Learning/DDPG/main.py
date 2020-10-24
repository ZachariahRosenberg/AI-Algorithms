import gym, os, random
import numpy      as np
import tensorflow as tf
from tqdm      import tqdm
from DDPG      import Agent
from utilities import plot_points

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(env, episodes=500, max_steps=500, eps_decay=.99,
         actor_lr=10**-6, critic_lr=10**-3, gamma=.9, 
         base_nodes=64, batch_size=128,theta=.4, sigma=.25):

	with tf.Session() as sess:

		# Initialize environment and constants
		input_dim   = env.state_dim   
		output_dim  = env.action_dim  
		action_high = env.action_high 
		action_low  = env.action_low 

		# Create DDPG Agent
		agent = Agent(input_dim, output_dim, action_high, action_low, 
		              actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, 
		              base_nodes=base_nodes, eps_decay=eps_decay,
		              batch_size=batch_size,theta=theta, sigma=sigma,
		              sess=sess)

		sess.run(tf.global_variables_initializer())
		agent.actor.update_target_network()
		agent.critic.update_target_network()

		# Prepare for episodes
		c_losses, rewards, actions, Qs, states = [np.array([]) for i in range(5)]

		for e in tqdm(range(episodes)):

			# Reset episode
			state = env.reset()
			state = np.reshape(state, (-1, len(state)))
			agent.noise.reset()

			done         = False
			step_count   = 0
			total_reward = 0

			while not done and step_count < max_steps:

				# Action
				action = agent.act(state)
				next_state, reward, done = env.step(action)
				next_state = np.reshape(next_state, (-1, len(next_state)))

				# Learn
				c_loss = agent.learn(state, action, reward, done, next_state)
				
				# Save results
				c_losses = np.append(c_losses, c_loss)
				actions  = np.append(actions, action)
				states   = np.append(states, state[0])
				Qs       = np.append(Qs, agent.critic.predict(state, action))
				
				# Loop
				state         = next_state
				step_count   += 1
				total_reward += reward

			# Reduce exploration
			if agent.eps > agent.min_eps:
				agent.eps *= agent.eps_decay

			rewards = np.append(rewards, total_reward)


		return rewards, c_losses, actions, Qs


# ***** This section is my hack for GridSearch and Plotting results *****

# Eps Decay: 
# .995 = 919
# .99  = 459
# .97  = 152
# .95  = 90
# .90  = 44

# Used for graphs
rewards, labels_rewards, c_losses, \
labels_critic, actions, Qs = [[] for i in range(6)]

# Grid search entries
labels     = ['quick', 'mid', 'slow']
a_lr       = [10**-3, 10**-6, 10**-8]
c_lr       = [10**-2, 10**-4, 10**-7] 
base_nodes = [300,      300,    350] 
gamma      = [.99,      .99,    .99]
batch_size = [128,      128,    128] 
theta      = [.15,      .15,    .15] 
sigma      = [.30,      .30,    .30]


env   = gym.make('pendulum-v0')

for lbl, alr, clr, base, thta, sgm, gma, bsize in zip(labels, a_lr, c_lr, base_nodes, theta, sigma, gamma, batch_size):

	print('Run {} of {}'.format(count, len(labels)))

	reward, c_loss, action, Q = main(env, 2000, max_steps=1000, base_nodes=base,
                                     actor_lr=alr, critic_lr=clr, batch_size=bsize,
                               		 theta=thta, sigma=sgm, gamma=gma,
                                   	 eps_decay = .99)

	rewards.append(reward)
	labels_rewards.append('{}'.format(lbl))
	
	c_losses.append(c_loss)
	labels_critic.append('{}'.format(lbl))

	actions.append(action)
	Qs.append(Q)
	reports.append(report)

plot_points(c_losses[-100:],
            labels_critic,
            title='Critic Loss',
            regression=True, show_plot=True)

plot_points(rewards,
        labels_rewards,
        title='Rewards',
        regression=True, show_plot=True)

