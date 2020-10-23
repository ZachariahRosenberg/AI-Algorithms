# -- Grid Search for OU Noise --

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import copy

# Default OU Noise 
class OUNoise:
    """Ornstein-Uhlenbeck process."""

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
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


print('getting started')

mu = 0
theta = 0.1
sigma = 0.2
noise1 = OUNoise(1, mu, .15, .5)
noise2 = OUNoise(1, mu, .15, .5)
noise3 = OUNoise(1, mu, .15, .5)

print('getting points')
points1 = []
for _ in range(10000):
	point = noise1.sample()
	points1.append(point[0])

points2 = []
for _ in range(10000):
	point = noise2.sample()
	points2.append(point[0])

points3 = []
for _ in range(10000):
	point = noise3.sample()
	points3.append(point[0])

print('got points, plotting')
num_bins = 10

plt.hist(points1, num_bins, label='.15 .2', histtype='step')
plt.hist(points2, num_bins, label='.15 .3', histtype='step')
plt.hist(points3, num_bins, label='.15 .4', histtype='step')

plt.legend(loc='upper right')
print('showing')
plt.show()

