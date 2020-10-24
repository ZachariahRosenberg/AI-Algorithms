import gym
import numpy      as np
import tensorflow as tf
from tqdm import tqdm

class Helpers:

    def discount_cumsum(rewards, discount):
        '''
        Method generates a discounted cumulative sum from a reward trajectory
        [1,2,3] -> [3*d + 2*d + 1*d, 2*d + 1*d, 1*d]
        '''
        result = []
        for i in range(len(rewards)):
            d_cummulative_sum = sum([discount * x for x in rewards[i:]])
            result.append(d_cummulative_sum)

        return result

    def mlp(x, hidden_sizes, activation=tf.nn.relu, output_activation=None):
        '''
        Cleanest dynamic generation of dense layered NN I've ever seen
        '''
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


class PGBuffer:

    def __init__(self, obs_dim, gamma=0.99, lam=0.95):

        self.obs_dim  = obs_dim

        self.obs_buf  = np.array([])
        self.act_buf  = np.array([])
        self.rew_buf  = np.array([])
        self.val_buf  = np.array([])
        self.adv_buf  = np.array([])
        self.ret_buf  = np.array([])

        self.gamma    = gamma
        self.lam      = lam
        self.idx_s    = 0
        self.idx_e    = 0

    def store(self, obs, act, rew, val):

        self.obs_buf  = np.append(self.obs_buf, obs)
        self.act_buf  = np.append(self.act_buf, act)
        self.rew_buf  = np.append(self.rew_buf, rew)
        self.val_buf  = np.append(self.val_buf, val)

        self.idx_e += 1

    def finish_path(self):

        path_slice = slice(self.idx_s, self.idx_e)
        rews = np.append(self.rew_buf[path_slice], 0)
        vals = np.append(self.val_buf[path_slice], 0)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf = np.append(self.adv_buf, Helpers.discount_cumsum(deltas, self.gamma * self.lam))

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf = np.append(self.ret_buf, Helpers.discount_cumsum(rews, self.gamma)[:-1])

        self.idx_s = self.idx_e

    def get(self):
        # Normalize the advantages
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        # Reshape observation
        shaped_obs = np.reshape(self.obs_buf, (-1,self.obs_dim))

        return [shaped_obs, self.act_buf, self.adv_buf, self.ret_buf]

    def reset(self):

        self.obs_buf  = np.array([])
        self.act_buf  = np.array([])
        self.rew_buf  = np.array([])
        self.val_buf  = np.array([])
        self.adv_buf  = np.array([])
        self.ret_buf  = np.array([])
        self.idx_e    = 0
        self.idx_s    = 0


def vpg(env_name='CartPole-v0', hidden_sizes=[64,64], pi_lr=1e-2, v_lr=1e-2, 
        epochs=100, batch_size=32, render=True, train_pi_iters=80, train_v_iters=80):

    # env constants
    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    seed = 42
    tf.set_random_seed(seed)
    np.random.seed(seed)
    buf = PGBuffer(obs_dim)

    # placeholders
    obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    act_ph = tf.placeholder(tf.int32,   shape=(None,))
    rew_ph = tf.placeholder(tf.float32, shape=(None,))
    adv_ph = tf.placeholder(tf.float32, shape=(None,))
    ret_ph = tf.placeholder(tf.float32, shape=(None,))

    # Actor
    with tf.variable_scope('pi'):
        logits    = Helpers.mlp(obs_ph, hidden_sizes=hidden_sizes+[act_dim], activation=tf.nn.relu, output_activation=None)
        pi        = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=1)
        logp      = tf.reduce_sum(tf.one_hot(act_ph, depth=act_dim) * tf.nn.log_softmax(logits), axis=1)

        pi_loss   = -tf.reduce_mean(logp * adv_ph)
        train_pi  = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)

    # Critic
    with tf.variable_scope('v'):
        v = tf.squeeze(Helpers.mlp(obs_ph, hidden_sizes=hidden_sizes+[1], activation=tf.tanh, output_activation=None), axis=1)

        v_loss  = tf.reduce_mean((ret_ph - v) ** 2)
        train_v = tf.train.AdamOptimizer(learning_rate=v_lr).minimize(v_loss)

    # Init sessions
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())

    # -------- Epoch -------- #
    for epoch in tqdm(range(epochs)):

        ep_rews = []
        finished_rendering_this_epoch = False

        # -------- Start Batch -------- #
        for episode in range(batch_size):

            total_episode_reward = 0
            obs, done, rew = env.reset(), False, 0

            # -------- Play -------- #
            while not done:

                if render and not finished_rendering_this_epoch:
                    env.render()

                a_t, v_t  = s.run([pi, v], {obs_ph: [obs]})
                obs, rew, done, _ = env.step(a_t[0])
                buf.store(obs, a_t, rew, v_t)

                total_episode_reward += rew

                if done:
                    ep_rews.append(total_episode_reward)
                    finished_rendering_this_epoch = True
                    buf.finish_path()

        # -------- Training -------- #
        inputs = {k:v for k,v in zip([obs_ph, act_ph, adv_ph, ret_ph], buf.get())}

        # Policy gradient step
        for _ in range(train_pi_iters):
            _, p_loss = s.run([train_pi, pi_loss], feed_dict=inputs)
        # Value function learning
        for _ in range(train_v_iters):
            _, v_l = s.run([train_v, v_loss], feed_dict=inputs)

        print(f'Epoch {epoch}, rewards: {sum(ep_rews)/len(ep_rews)}')

        # -------- Reset -------- #
        buf.reset()


if __name__ == '__main__':

    envs = ['MountainCar-v0', 'CartPole-v0']

    params = {
        'env_name'      : envs[1],
        'hidden_sizes'  : [64,64],
        'pi_lr'         : 1e-2,
        'v_lr'          : 1e-2,
        'epochs'        : 100,
        'batch_size'    : 32,
        'render'        : False,
        'train_pi_iters': 1,
        'train_v_iters' : 80
    }

    vpg(**params)