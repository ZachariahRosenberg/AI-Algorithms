import gym, keras, random, seaborn
import numpy    		 as np
import matplotlib.pyplot as plt
from tqdm               import tqdm
from collections        import deque
from keras.layers       import Input, Dense, Dropout, BatchNormalization
from keras.models       import Model
from keras.optimizers   import Adam
from keras.initializers import TruncatedNormal

# Plots Rewards & Losses
class Plot():

	def __init__(self):

		plt.ion()
		self.fig          = plt.figure()

		self.ax_rewards    = self.fig.add_subplot(211)
		self.plot_rewards, = self.ax_rewards.plot([0],[0])
		self.ax_losses     = self.fig.add_subplot(212)
		self.plot_losses,  = self.ax_losses.plot([0],[0])

		self.ax_rewards.set_title('Rewards')
		self.ax_losses .set_title('Losses')

		self.fig.canvas.draw()
		plt.show(block=False)

	def update(self, new_rewards, new_losses):

		# Draw Rewards
		self.plot_rewards.set_xdata(np.append(self.plot_rewards.get_xdata(), [len(self.plot_rewards.get_xdata())]))
		self.plot_rewards.set_ydata(np.append(self.plot_rewards.get_ydata(), new_rewards))
		self.ax_rewards.relim()
		self.ax_rewards.autoscale_view(True,True,True)

		# Draw Losses
		self.plot_losses.set_xdata(np.append(self.plot_losses.get_xdata(), [len(self.plot_losses.get_xdata())]))
		self.plot_losses.set_ydata(np.append(self.plot_losses.get_ydata(), new_losses))
		self.ax_losses.relim()
		self.ax_losses.autoscale_view(True,True,True)

		plt.pause(.0001)
		plt.draw()


class DQN_Agent():

	def __init__(self, options):

		# Q Hyperparameters
		self.batch_size = options.get('batch_size', 32)
		self.n_batches  = options.get('n_batches', 1)
		self.gamma      = options.get('gamma', .95)
		self.eps        = options.get('eps', 1.0)
		self.min_eps    = options.get('min_eps', .1)
		self.eps_decay  = options.get('eps_decay', .95)

		# Model Hyperparameters
		self.lr         = options.get('lr', 10e-3)
		self.base_nodes = options.get('base_nodes', 32)
		self.input_dim  = options['input_dim']
		self.output_dim = options['output_dim']
		self.d_rate     = options.get('d_rate', .2)
		
		self.memory     = deque(maxlen=2000)
		self.model      = self.build_model()

	def build_model(self):

		inputs  = Input(shape=(self.input_dim,))
		hidden  = Dense(self.base_nodes * 1, activation='relu', 
		                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42), 
		                bias_initializer  =TruncatedNormal(mean=0.0, stddev=0.05, seed=42))(inputs)
		hidden  = Dense(self.base_nodes * 2, activation='relu',
		                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42), 
		                bias_initializer  =TruncatedNormal(mean=0.0, stddev=0.05, seed=42))(hidden)
		outputs = Dense(self.output_dim, activation='linear',
		                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42), 
		                bias_initializer  =TruncatedNormal(mean=0.0, stddev=0.05, seed=42))(hidden)

		model = Model(inputs=inputs, outputs=outputs)
		model.compile(optimizer=Adam(lr=self.lr), loss='mse')

		return model

	def act(self, state):

		if np.random.rand() < self.eps:
			action = np.random.rand(self.output_dim)
		else:
			action = self.model.predict(state)[0]

		return np.argmax(action)

	def remember(self, state, action, reward, next_state, done):

		self.memory.append((state, action, reward, next_state, done))

	def learn(self):

		# Create batches
		batch_size = np.minimum(self.batch_size, len(self.memory))
		# TODO: try using prioritized memory recall
		batches = np.array([random.sample(self.memory, batch_size) for i in range(self.n_batches)])

		# Split batches into components
		states   = np.vstack([b[0] for batch in batches for b in batch])
		actions  = np.array ([b[1] for batch in batches for b in batch])
		rewards  = np.array ([b[2] for batch in batches for b in batch]).reshape(-1, 1)
		n_states = np.vstack([b[3] for batch in batches for b in batch])
		dones    = np.array ([int(b[4]) for batch in batches for b in batch]).reshape(-1, 1)

		# Get predictions on batches
		predictions = self.model.predict(states)
		n_actions   = self.model.predict(n_states)

		# Learn - Q() = r + gma * (Q(s',a))
		targets     = rewards + self.gamma * np.vstack(np.amax(n_actions, axis=1)) * (1 - dones)

		# Correct the predictions with targets
		# TODO: try switching to advantage
		t_actions = []
		for state, action, prediction, target in zip(states, actions, predictions, targets):
			prediction[action] = target
			t_actions.append(prediction)

		# Fit
		loss = self.model.fit(states, 
							  np.array(t_actions), 
							  batch_size=batch_size,
							  verbose=0,
							  epochs=1).history['loss']

		return loss


def main(env, episodes=1000, max_steps=1000, agent_options=None, env_render=False, show_plot=False, reward_function=None):

	agent           = DQN_Agent(agent_options)
	init_weights    = agent.model.get_weights()
	rewards, losses = ([], [])
	plot = Plot() if show_plot == True else None

	for e in tqdm(range(episodes)):

		# Reset episode
		state      = env.reset()[None, :]
		done       = False
		step_count = 0
		e_reward   = 0
		e_loss     = 0

		# Begin Episode
		while done != True and step_count < max_steps:

			if env_render is True:
				env.render()

			# Action
			action = agent.act(state)
			next_state, reward, done, __ = env.step(action)
			next_state = next_state[None, :]

			# Custom reward function
			if reward_function is not None:
				reward = reward_function(reward, done)

			# Remember state
			agent.remember(state, action, reward, next_state, done)

			# Learn
			losses = agent.learn()
			
			# Loop
			state       = next_state
			step_count += 1
			e_reward   += reward
			e_loss     += np.mean(losses)

		# Reduce exploration
		if agent.eps > agent.min_eps:
			agent.eps *= agent.eps_decay

		# Save rewards/losses
		rewards.append(e_reward)
		losses .append(e_loss)
		print('Eps: {}, ep reward: {}, avg t100: {}'.format(round(agent.eps,2), e_reward, round(np.mean(rewards[:-100]),2)))

		# Plot rewards/losses
		if plot is not None:
			plot.update([e_reward], [e_loss])

	final_weights = agent.model.get_weights()

	return rewards, losses, init_weights, final_weights


if __name__ == "__main__":

	env       = gym.make('CartPole-v1')
	episodes  = 250

	agent_options = {
		'base_nodes':  24,
		'lr'        : .001,
		'n_batches' :  32,
		'batch_size':  1,
		'gamma'     : .95,
		'eps'       : 1.0,
		'min_eps'   :  0,
		'eps_decay' : .99,
		'd_rate'    : .3,
		'input_dim' : env.observation_space.shape[0],
		'output_dim': env.action_space.n 
	}

	# Currently tuned for Cartpole
	def reward_engineering(reward, done):
		return reward if not done else -10

	# Plot final results
	def plot_final(old_weights, new_weights, rewards, losses):
		seaborn.set()
		plt.ioff()
		plt.clf()

		plt.subplot(3, 3, 1)
		plt.plot(rewards)
		plt.ylabel('rewards')

		plt.subplot(3, 3, 4)
		plt.plot(losses)
		plt.ylabel('losses')

		plt.subplot(3, 3, 2)
		seaborn.heatmap(old_weights[0])
		plt.subplot(3, 3, 5)
		seaborn.heatmap(old_weights[2])
		plt.subplot(3, 3, 8)
		seaborn.heatmap(old_weights[4])
		plt.ylabel('old weights')

		plt.subplot(3, 3, 3)
		seaborn.heatmap(new_weights[0])
		plt.subplot(3, 3, 6)
		seaborn.heatmap(new_weights[2])
		plt.subplot(3, 3, 9)
		seaborn.heatmap(new_weights[4])
		plt.ylabel('new weights')

		plt.show()

	rewards, losses, old_weights, new_weights = main(env, episodes, agent_options=agent_options, env_render=False, show_plot=True, reward_function=reward_engineering)

	plot_final(old_weights, new_weights, rewards, losses)