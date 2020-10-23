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


class PPOBuffer:

    def __init__(self, obs_dim, gamma=0.99, lam=0.97):

        self.obs_dim   = obs_dim

        self.obs_buf   = np.empty((0, self.obs_dim))
        self.act_buf   = np.array([])
        self.rew_buf   = np.array([])
        self.val_buf   = np.array([])
        self.adv_buf   = np.array([])
        self.ret_buf   = np.array([])
        self.logp_buf  = np.array([])

        self.gamma     = gamma
        self.lam       = lam
        self.idx_s     = 0
        self.idx_e     = 0

    def store(self, obs, act, rew, val, logp):

        self.obs_buf  = np.append(self.obs_buf, obs, axis=0)
        self.act_buf  = np.append(self.act_buf, act)
        self.rew_buf  = np.append(self.rew_buf, rew)
        self.val_buf  = np.append(self.val_buf, val)
        self.logp_buf = np.append(self.logp_buf, logp)

        self.idx_e += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.idx_s, self.idx_e)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf = np.append(self.adv_buf, Helpers.discount_cumsum(deltas, self.gamma * self.lam))

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf = np.append(self.ret_buf, Helpers.discount_cumsum(rews, self.gamma)[:-1])

        self.idx_s = self.idx_e

    def get(self):
        # Normalize the advantages
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)

        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]

    def reset(self):

        self.obs_buf  = np.empty((0, self.obs_dim))
        self.act_buf  = np.array([])
        self.rew_buf  = np.array([])
        self.val_buf  = np.array([])
        self.adv_buf  = np.array([])
        self.ret_buf  = np.array([])
        self.logp_buf = np.array([])
        self.idx_e    = 0
        self.idx_s    = 0


def ppo(env_name='CartPole-v0', hidden_sizes=[64,64], pi_lr=3e-4, v_lr=1e-3, clip_ratio=0.2,
        max_kl=0.01, epochs=100, render=True, train_pi_iters=80, train_v_iters=80,
        steps_per_epoch=4000):

    # env constants
    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    seed = 42
    tf.set_random_seed(seed)
    np.random.seed(seed)
    buf = PPOBuffer(obs_dim)

    # placeholders
    obs_ph      = tf.placeholder(tf.float32, shape=(None, obs_dim))
    act_ph      = tf.placeholder(tf.int32,   shape=(None,))
    rew_ph      = tf.placeholder(tf.float32, shape=(None,))
    adv_ph      = tf.placeholder(tf.float32, shape=(None,))
    ret_ph      = tf.placeholder(tf.float32, shape=(None,))
    logp_old_ph = tf.placeholder(tf.float32, shape=(None,))

    # Actor
    with tf.variable_scope('pi'):
        logits   = Helpers.mlp(obs_ph, hidden_sizes=hidden_sizes+[act_dim], activation=tf.nn.relu, output_activation=None)
        pi       = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=1)
        logp     = tf.reduce_sum(tf.one_hot(act_ph, depth=act_dim) * tf.nn.log_softmax(logits), axis=1) # Previous Actions
        logp_pi  = tf.reduce_sum(tf.one_hot(pi,     depth=act_dim) * tf.nn.log_softmax(logits), axis=1) # Hypothetical Actions

        # PPO objectives
        ratio       = tf.exp(logp - logp_old_ph)
        clipped_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
        pi_loss     = -tf.reduce_mean(tf.minimum(ratio * adv_ph, clipped_adv))
        train_pi    = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)

        # PPO Used for early stopping during training
        approx_kl = tf.reduce_mean(logp_old_ph - logp)

    # Critic
    with tf.variable_scope('v'):
        v = tf.squeeze(Helpers.mlp(obs_ph, hidden_sizes=hidden_sizes+[1], activation=tf.tanh, output_activation=None), axis=1)

        v_loss  = tf.reduce_mean((ret_ph - v) ** 2)
        train_v = tf.train.AdamOptimizer(learning_rate=v_lr).minimize(v_loss)

    # Init sessions
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())

    # Update Function
    def update(feed_ph, sess, max_kl=0.01, train_pi_iters=80, train_v_iters=80):
        inputs = {k:v for k,v in zip(feed_ph, buf.get())}

        # Policy gradient step
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            # Early stopping
            kl = np.absolute(np.mean(kl))
            if kl > 1.5 * max_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break

        # Value function learning
        for _ in range(train_v_iters):
            _ = sess.run([train_v], feed_dict=inputs)

    # -------- Epoch -------- #
    for epoch in tqdm(range(epochs)):

        obs, rew, done = env.reset(), 0, False
        total_episode_reward, epoch_rews = 0, []
        finished_rendering_this_epoch    = False

        # -------- Start Batch -------- #
        for step in range(steps_per_epoch):

            if render and not finished_rendering_this_epoch:
                env.render()

            act, v_t, logp_t = s.run([pi, v, logp_pi], feed_dict={obs_ph: obs.reshape(1,-1)})
            buf.store([obs], act, rew, v_t, logp_t)
            
            obs, rew, done, _ = env.step(act[0])
            total_episode_reward += rew

            if done or (step==steps_per_epoch-1):
                last_val = rew if done else s.run(v, feed_dict={obs_ph: obs.reshape(1,-1)})
                buf.finish_path(last_val)
            
                epoch_rews.append(total_episode_reward)
            
                obs, rew, done       = env.reset(), 0, False
                total_episode_reward =0

                finished_rendering_this_epoch = True

        update([obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph], s, max_kl=max_kl, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters)
        buf.reset()
        print(f'Epoch {epoch}, rewards: {sum(epoch_rews)/len(epoch_rews)}')


if __name__ == '__main__':

    envs = ['MountainCar-v0', 'CartPole-v0', 'Pong-ram-v0']

    params = {
        'env_name'      : envs[1],
        'hidden_sizes'  : [64,64],
        'pi_lr'         : 3e-4,
        'v_lr'          : 1e-3,
        'clip_ratio'    : 0.2,
        'max_kl'        : .01,
        'epochs'        : 250,
        'render'        : False,
        'train_pi_iters': 80,
        'train_v_iters' : 80
    }

    ppo(**params)