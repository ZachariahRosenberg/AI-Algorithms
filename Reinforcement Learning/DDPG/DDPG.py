import tflearn
import numpy as np
import tensorflow as tf
from utilities import OUNoise, ReplayBuffer

'''
Actor-Critic model based on model by
Patrick Emami, https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
'''

class Agent:

    def __init__(self, input_dim, output_dim, action_high, action_low, 
                     actor_lr=10**-3, critic_lr=10**-2, gamma=.99, 
                     batch_size=64, theta=.15, sigma=.3, sess=None,
                     base_nodes=64, tau=.001, eps_decay=.99):


            self.action_high = action_high
            self.action_low  = action_low
            self.batch_size  = batch_size
            self.memory      = ReplayBuffer()
            
            self.eps       = 1
            self.eps_decay = eps_decay
            self.min_eps   = .25
            self.gamma     = gamma

            self.mu    = 0
            self.theta = theta
            self.sigma = sigma
            self.noise = OUNoise(output_dim, self.mu, self.theta, self.sigma)


            with tf.variable_scope("actor", reuse=tf.AUTO_REUSE):
                self.actor  = Actor(sess, input_dim, output_dim, action_high, action_low, 
                                           lr=actor_lr, tau=tau, batch_size=batch_size, base_nodes=base_nodes)
            
            with tf.variable_scope("critic", reuse=tf.AUTO_REUSE):
                self.critic = Critic(sess, input_dim, output_dim, 
                                            lr=critic_lr, tau=tau, base_nodes=base_nodes)


    def act(self, state):

        noise  = self.noise.sample() * self.eps
        action = self.actor.predict(state) 

        action = np.add(action, noise)

        return np.clip(action, self.action_low, self.action_high)


    def learn(self, state, action, reward, done, next_state):

        self.memory.add(state, action, reward, done, next_state)

        # Get batch of experiences
        batch_size  = np.min([self.batch_size, len(self.memory)])
        experiences = self.memory.sample(batch_size)

        states  = np.vstack([e[0] for e in experiences if e is not None])
        actions = np.array ([e[1] for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.critic.action_dim)
        rewards = np.array ([e[2] for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones   = np.array ([e[3] for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e[4] for e in experiences if e is not None])

        # Generate SARSA
        next_actions = self.actor.predict_target(next_states)
        next_Qs      = self.critic.predict_target(next_states, next_actions)

        # Train Critic
        actual_Qs   = rewards + self.gamma * next_Qs * (1 - dones)
        loss_critic = self.critic.train(states, actions, actual_Qs, batch_size=batch_size)[0]   

        # Train Actor
        action_gradients = self.critic.get_action_gradients([states, self.actor.predict(states)])
        self.actor.train([states, action_gradients[0]])

        # Soft-update target models
        self.actor.update_target_network()
        self.critic.update_target_network()
        
        return loss_critic


class Actor(object):

    def __init__(self, sess, state_dim, action_dim, action_high, action_low, lr=10**-3, tau=.001, batch_size=64, base_nodes=32):

        self.lr   = lr
        self.tau  = tau
        self.sess = sess
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.action_high = action_high
        self.action_low  = action_low
        self.batch_size  = batch_size
        self.base_nodes  = base_nodes

        # Actor Network
        self.inputs, self.out = self.build_model()
        self.weights = tf.trainable_variables(scope='actor')

        # Target Network
        self.target_inputs, self.target_out = self.build_model()
        self.target_weights = tf.trainable_variables(scope='actor')[len(self.weights):]

        # Soft update target network
        self.update_target_weights = \
            [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
            for i in range(len(self.target_weights))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])


        self.unnormalized_actor_gradients = tf.gradients(self.out, self.weights, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.weights))


    def build_model(self):

        inputs = tflearn.input_data(shape=[None, self.state_dim])

        net = tflearn.fully_connected(inputs, self.base_nodes)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.dropout(net, .8)

        net = tflearn.fully_connected(net, self.base_nodes* 2)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.dropout(net, .8)

        net = tflearn.fully_connected(net, self.base_nodes)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        out = tflearn.fully_connected(net, self.action_dim, activation='tanh', 
                                      weights_init=tflearn.initializations.uniform(minval=-.25, maxval=.25))

        return inputs, out

    def train(self, inputs):

        states, gradients = inputs

        return self.sess.run([self.optimize], feed_dict={
            self.inputs: states,
            self.action_gradient: gradients
        })

    def predict(self, inputs):

        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):

        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):

        self.sess.run(self.update_target_weights)


class Critic(object):

    def __init__(self, sess, state_dim, action_dim, lr=10**-2, base_nodes=64, tau=.001):
        
        self.lr    = lr
        self.tau   = tau
        self.sess  = sess
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.base_nodes = base_nodes

        # Create the critic network
        self.inputs, self.action, self.out = self.build_model()
        self.weights = tf.trainable_variables(scope='critic')

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.build_model()
        self.target_weights = tf.trainable_variables(scope='critic')[len(self.weights):]

        # Soft update target net work
        self.update_target = \
            [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                for i in range(len(self.target_weights))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss     = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.action_grads = tf.gradients(self.out, self.action)


    def build_model(self):

        # State Path
        inputs = tflearn.input_data(shape=[None, self.state_dim])

        net_state = tflearn.fully_connected(inputs, self.base_nodes*2)
        net_state = tflearn.layers.normalization.batch_normalization(net_state)
        net_state = tflearn.activations.relu(net_state)
        net_state = tflearn.dropout(net_state, .8)

        t1 = tflearn.fully_connected(net_state, self.base_nodes)
        
        # Action Path
        action = tflearn.input_data(shape=[None, self.action_dim])
        t2   = tflearn.fully_connected(action, self.base_nodes)

        # Add the action tensor in the 2nd hidden layer
        net = tflearn.activation(tf.matmul(net_state, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        net = tflearn.dropout(net, .8)

        net = tflearn.fully_connected(net, self.base_nodes)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        out = tflearn.fully_connected(net, 1, weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
        
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value, batch_size=64):

        predicted_q_value = np.reshape(predicted_q_value, (batch_size, 1))

        return self.sess.run([self.loss, self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):

        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):

        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def get_action_gradients(self, inputs):

        states, actions = inputs

        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: states,
            self.action: actions
        })

    def update_target_network(self):

        self.sess.run(self.update_target)