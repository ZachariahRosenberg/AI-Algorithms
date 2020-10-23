import numpy as np
import pylab, copy, random
from scipy       import stats
from datetime    import datetime
from collections import deque

#Default ReplayBuffer
class ReplayBuffer:

    def __init__(self, buffer_size=10**5):

        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, done, next_state):

        self.memory.append([state, action, reward, done, next_state])

    def sample(self, batch_size=32):

        return random.sample(self.memory, k=batch_size)

    def __len__(self):

        return len(self.memory)

#Default OU-Noise
class OUNoise:

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# Scale [prev_min, prev_max] to scale [new_min, new_max] Used to scale actions
def scale_inputs(x, prev_min, prev_max, new_min, new_max):

	return ((new_max-new_min) * (x - prev_min))/(prev_max-prev_min) + new_min

# Graph Control Panel for pendulum
def graph_control(agent, rewards, positions, actions, Qs, c_losses, 
                  s1_low=-1., s1_high=1., s2_low=-8., s2_high=8., act_low=-2., 
                  act_high=2., show_plot=True):

	pylab.close()
	
	# Display plots.
	fig = pylab.figure(figsize=(11,6))
	ax0 = pylab.subplot2grid((3, 9), (0, 0), colspan=3, rowspan=3)
	ax1 = pylab.subplot2grid((3, 9), (0, 3), colspan=3, rowspan=1)
	ax2 = pylab.subplot2grid((3, 9), (1, 3), colspan=3, rowspan=1)
	ax3 = pylab.subplot2grid((3, 9), (2, 3), colspan=3, rowspan=1)
	ax4 = pylab.subplot2grid((3, 9), (0, 6), colspan=3, rowspan=1)
	ax5 = pylab.subplot2grid((3, 9), (1, 6), colspan=3, rowspan=2)
            
	# Sample the state space to get some action values.
	ax0.set_title('Policy')
	n = 50

	s1_sample  = np.tile(np.linspace(s1_low, s1_high, n), n)
	s2_sample  = np.repeat(np.linspace(s2_low, s2_high, n), n)
	act_Sample = [agent.actor.predict_target(np.array([[s1, s2]])) for s1, s2 in zip(s1_sample, s2_sample)]

	cLevels = [act_low, act_high]
	im = ax0.contourf(np.linspace(s1_low, s1_high, n), np.linspace(s2_low, s2_high, n), np.reshape(act_Sample, [n,n]), cmap='jet', vmin=act_low, vmax=act_high)
	fig.colorbar(im, ax=ax0, ticks=cLevels)

	ax1.set_title('Reward')
	ax1.plot(rewards)

	ax2.set_title('Position')         
	ax2.plot([0, len(positions)], [0.5,  0.5], c='k', linewidth=0.5) # Plot goal position
	ax2.plot([0, len(positions)], [-0.5,-0.5], c='k', linewidth=0.5) # Plot start position
	ax2.plot(positions, label='Position')
	ax2.set_ylim(s1_low, s1_high)
 
	ax3.set_title('Action')           
	ax3.plot([0, len(actions)], [0,0], c='k', linewidth=0.5)
	ax3.plot(actions, label='Action')
	ax3.set_ylim(act_low, act_high)
 
	ax4.set_title('Qs')           
	ax4.plot(np.array(Qs), label='Q')

	ax5.set_title('Critic Loss')
	ax5.plot(np.array(c_losses), label='Critic Loss')

	pylab.savefig('graphs/graph_cntrl_'+str(datetime.now())+'.png')
    
	# pylab.ion()
	# pylab.show(block=False)
	# pylab.pause(0.001)

# Plot line graph with optional regression
def plot_points(points, labels, title='Plot', regression=False, show_plot=True):

	for i, data in enumerate(zip(points, labels)):
		point, label = data
		point = np.array(point).flatten()
		x = np.arange(len(point))
		slope, intercept, r_value, p_value, std_err = stats.linregress(x, point)
		pylab.subplot(len(labels),1,i+1)
		pylab.title(title+' - '+label)
		#pylab.legend(loc="upper right")
		pylab.plot(x, point, label=label, ls='-')
		if regression:
			line = slope*x+intercept
			pylab.plot(x, line,  label='reg:'+label)

	pylab.savefig('graphs/'+title+'_'+str(datetime.now())+'.png')
	if show_plot:
		pylab.show()

	pylab.gcf().clear()

# Graph for quadcopter
def graph_quad(results, rewards, critic_losses):

	pylab.close()

	fig = pylab.figure(figsize=(11,6))
	ax0 = pylab.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=1) #xyz positions
	ax1 = pylab.subplot2grid((3, 6), (1, 0), colspan=3, rowspan=2) #xyz velocities
	ax2 = pylab.subplot2grid((3, 6), (0, 3), colspan=3, rowspan=1) #rotor speeds
	ax3 = pylab.subplot2grid((3, 6), (1, 3), colspan=3, rowspan=1) #rewards
	ax4 = pylab.subplot2grid((3, 6), (2, 3), colspan=3, rowspan=1) #critic loss

	ax0.set_title('pos')
	ax0.plot(results['time'], results['x'], label='x')
	ax0.plot(results['time'], results['y'], label='y')
	ax0.plot(results['time'], results['z'], label='z')
	ax0.legend()

	ax1.set_title('vel')
	ax1.plot(results['time'], results['x_velocity'], label='dx')
	ax1.plot(results['time'], results['y_velocity'], label='dy')
	ax1.plot(results['time'], results['z_velocity'], label='dz')
	ax1.legend()

	ax2.set_title('rotors')
	ax2.plot(results['time'], results['rotor_speed1'], label='r1 r/sec')
	ax2.plot(results['time'], results['rotor_speed2'], label='r2 r/sec')
	ax2.plot(results['time'], results['rotor_speed3'], label='r3 r/sec')
	ax2.plot(results['time'], results['rotor_speed4'], label='r4 r/sec')
	ax2.legend()

	ax3.set_title('Reward')
	ax3.plot(rewards, label='Rewards')

	ax4.set_title('Critic Loss')
	ax4.plot(np.array(critic_losses), label='Critic Loss')

	pylab.savefig('graphs/quad_'+str(datetime.now())+'.png')

	pylab.ion()
	pylab.show(block=False)
	pylab.pause(0.001)