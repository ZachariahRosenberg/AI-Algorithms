import gym, datetime
import numpy      as np
import torch      as t
import torch.nn   as nn
import matplotlib.pyplot as plt

from collections    import OrderedDict
from tqdm           import tnrange
from torch.autograd import Variable

device = t.device('cuda')
seed   = 42
t.manual_seed(seed)
np.random.seed(seed)

# ------ helper functions ------

def discount_cumsum(rewards, discount):
    result = []
    for i in range(len(rewards)):
        d_cummulative_sum = sum([x * discount**(idx) for idx, x in enumerate(rewards[i:])])
        result.append(d_cummulative_sum)
    return result

# ------ Buffer ------

class PPOBuffer:
    
    def __init__(self, obs_dim, gamma=0.99, lam=0.97):
        
        self.obs_dim   = obs_dim
        self.gamma     = gamma
        self.lam       = lam
        
        self.reset()
    
    def store(self, obs, act, rew, val, logp):
        
        self.obs_buf  = np.append(self.obs_buf,  obs, axis=0)
        self.act_buf  = np.append(self.act_buf,  act)
        self.rew_buf  = np.append(self.rew_buf,  rew)
        self.val_buf  = np.append(self.val_buf,  val)
        self.logp_buf = np.append(self.logp_buf, logp)

        self.idx_e += 1
    
    def finish_path(self, last_val=0):

        path_slice = slice(self.idx_s, self.idx_e)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]  # TD
        advs   = discount_cumsum(deltas, self.gamma * self.lam) # Monte Carlo
        self.adv_buf = np.append(self.adv_buf, advs)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf = np.append(self.ret_buf, discount_cumsum(rews, self.gamma)[:-1])

        self.idx_s = self.idx_e
        
    def get(self):

        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]
    
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

# ------ Model ------

class PPOModel(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_size=64, pi_lr=3e-4, v_lr=1e-3, 
                 clip_ratio=.2, max_kl=0.01, train_pi_iters=80, train_v_iters=80,
                 categorical=True, buffer_kwargs=None):
        super(PPOModel, self).__init__()
        
        # env
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.categorical = categorical
        
        if categorical:
            self.act = self.act_categorical
        else:
            self.log_std = nn.Parameter(-0.5 * t.ones(act_dim, dtype=t.float32))
            self.act    =  self.act_gaussian
        
        # params
        self.max_kl         = max_kl
        self.clip_ratio     = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters  = train_v_iters
        
        # Build models
        self.actor  = self.build_actor_model(hidden_size)
        self.critic = self.build_critic_model(hidden_size)
        self.a_op   = t.optim.Adam(self.actor.parameters(),  lr = pi_lr)
        self.c_op   = t.optim.Adam(self.critic.parameters(), lr = v_lr)
        
        # buffer
        self.buf = PPOBuffer(obs_dim, **buffer_kwargs)
        
        # plot material
        self.kl_hist     = []
        self.p_loss_hist = []
        self.v_loss_hist = []
        self.move_hist   = []
    
    def build_actor_model(self, h_dim):
        
        model = nn.Sequential(OrderedDict({
            'a_input': nn.Linear(self.obs_dim, h_dim), # input
            'a_relu1': nn.ReLU(),
            'a_h1'   : nn.Linear(h_dim, h_dim),        # hidden 1
            'a_relu2': nn.ReLU(),
            'a_h2'   : nn.Linear(h_dim, h_dim),        # hidden 2
            'a_relu3': nn.ReLU(),
            'a_out'  : nn.Linear(h_dim, self.act_dim)  # output  
        }))
        
        return model
    
    def build_critic_model(self, h_dim):
        
        model = nn.Sequential(OrderedDict({
            'c_input': nn.Linear(self.obs_dim, h_dim), # input
            'c_tanh1': nn.Tanh(),
            'c_h1'   : nn.Linear(h_dim, h_dim),        # hidden 1
            'c_tanh2': nn.Tanh(),
            'c_h2'   : nn.Linear(h_dim, h_dim),        # hidden 2
            'c_tanh3': nn.Tanh(),
            'c_out'  : nn.Linear(h_dim, 1)             # output  
        }))
        
        return model
    
    def act_categorical(self, obs):
        
        logits = self.actor(obs)
        policy = t.distributions.Categorical(logits=logits)
        pi     = policy.sample()
        logpi  = policy.log_prob(pi).squeeze()
        
        return pi, logpi
    
    def act_gaussian(self, obs):
        
        mu      = self.actor(obs)
        policy  = t.distributions.Normal(mu, self.log_std.exp())
        pi      = policy.sample()
        logpi   = policy.log_prob(pi).sum(dim=1)

        return pi, logpi
    
    def act(self, obs):
        
        raise NotImplementedError
    
    def val(self, obs):
        
        logits = self.critic(obs)
        val    = t.squeeze(logits, dim=1)
        
        return val
    
    def update(self):
        
        buf = {k:t.from_numpy(v).float() for k,v in zip(['obs', 'acts', 'advs', 'rets', 'logps'], self.buf.get())}

        # Normalize Advantages
        buf['advs'] = (buf['advs'] - t.mean(buf['advs'])) / t.std(buf['advs'])
        
        # Reshape acts if continuous
        acts = buf['acts']
        if not self.categorical:
            acts = np.reshape(acts, (len(acts), 1))

        # For Plotting
        kls, p_ls, v_ls  = [], [], []
        
        # Policy gradient step
        for i in range(self.train_pi_iters):

            if self.categorical:
                logits      = self.actor(buf['obs'])
                policy      = t.distributions.Categorical(logits=logits)
                logps       = policy.log_prob(acts).squeeze()
                
            else:
                mu     = self.actor(buf['obs'])
                policy = t.distributions.Normal(mu, self.log_std.exp())
                logps  = policy.log_prob(acts).sum(dim=1)
                
            ratio       = (logps - buf['logps']).exp()
            clipped_adv = t.where(buf['advs'] > 0, (1 + self.clip_ratio) * buf['advs'], (1 - self.clip_ratio) * buf['advs'])
            pi_loss     = -(t.min(ratio * buf['advs'], clipped_adv)).mean()
            p_ls.append(pi_loss.item())
            
            # Early stopping
            kl = (buf['logps'] - logps).mean().item()
            kls.append(kl)
            if abs(kl) > self.max_kl:
                break

            self.a_op.zero_grad()
            pi_loss.backward()
            self.a_op.step()

        # Value function learning
        for _ in range(self.train_v_iters):
            v      = self.val(buf['obs'])
            v_loss = ((buf['rets'] - v) ** 2).mean()
            v_ls.append(v_loss.item())

            self.c_op.zero_grad()
            v_loss.backward()
            self.c_op.step()
        
        self.kl_hist.append(np.mean(kls))
        self.p_loss_hist.append(np.mean(p_ls))
        self.v_loss_hist.append(np.mean(v_ls))  


# ------ Params ------

# Environment
envs    = ['CartPole-v0', 'Pendulum-v0', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pong-ram-v0']
env     = gym.make(envs[4])

# Params
epochs          = 5000
games_per_epoch = 10
steps_per_epoch = games_per_epoch * 1400
render          = False

# Discrete or Continuous action dim
categorical = True if isinstance(env.action_space, gym.spaces.Discrete) else False
act_dim     = env.action_space.n if categorical else env.action_space.shape[0]
obs_dim     = env.observation_space.shape[0]

model_kwargs = {
    'pi_lr'         : 3e-4,
    'v_lr'          : 1e-3,
    'clip_ratio'    : 0.2, #.1,.2,.3
    'max_kl'        : 0.03, #0.003 to 0.03
    'train_pi_iters': 80,
    'train_v_iters' : 80,
    'hidden_size'   : 64,
    'categorical'   : categorical
}
buffer_kwargs = {
    'lam'  : 0.97,
    'gamma': 0.99
}

print(f'Environment is {"Discrete" if categorical else "Continuous"} with {act_dim} actions')
try: print(env.unwrapped.get_action_meanings())
except: pass

# ------ Play! ------

fig = plt.figure(figsize=(9, 4))

try:
    model = t.load('./ppo.pt')
    print('Model loaded successfully.')
except:
    model = PPOModel(obs_dim, act_dim, buffer_kwargs=buffer_kwargs, **model_kwargs)
    print('Model not found. Creating new.')

total_rews = []
bar = tnrange(epochs)
for epoch in bar:
    
    t.save(model, './ppo.pt')

    # Reset and prepare
    obs, rew, done = env.reset(), 0, False
    total_episode_reward, epoch_rews = 0, []
    finished_rendering_this_epoch    = False
    game_n = 0
    
    # -------- Start Trajectory -------- #
    for step in range(steps_per_epoch):
        
        if games_per_epoch is not None and game_n == games_per_epoch:
            break

        if render and not finished_rendering_this_epoch:
            env.render()
        
        # -------- Action -------- #
        obs         = t.tensor([obs], requires_grad=False).float()
        act, logp_t = model.act(obs)
        act         = [act.item()]
        logp_t      = [logp_t.item()]
        v_t         = [model.val(obs).item()]
        
        # -------- Store -------- #
        model.buf.store(obs.numpy(), act, rew, v_t, logp_t)
        obs, rew, done, _ = env.step(act[0] if categorical else act)
        
        total_episode_reward += rew

        # -------- End Game Action-------- #
        if done or (step==steps_per_epoch-1):
            obs      = t.tensor([obs], requires_grad=False).float()
            last_val = rew if done else model.val(obs).item()
            model.buf.finish_path(last_val)

            epoch_rews.append(total_episode_reward)
            bar.write(f'Game {game_n+1} of epoch: {epoch+1}, reward: {total_episode_reward:.2f}')
            game_n += 1
            
            # -------- Reset -------- #
            obs, rew, done       = env.reset(), 0, False
            total_episode_reward = 0
            finished_rendering_this_epoch = True

    
    # -------- Train and Log -------- #
    model.update()
    model.buf.reset()
    total_rews.append(sum(epoch_rews)/len(epoch_rews))
    bar.write(f'Epoch {epoch+1}, rewards: {(sum(epoch_rews)/len(epoch_rews)):.2f}')
    
    ax_rw = plt.subplot2grid((3, 3), (0, 0), rowspan=2,colspan=3)
    ax_rw.set_title('Rewards')
    ax_rw.plot(total_rews)

    ax_pl = plt.subplot2grid((3, 3), (2, 0))
    ax_pl.set_title('Policy Loss')
    ax_pl.plot(model.p_loss_hist)

    ax_vl = plt.subplot2grid((3, 3), (2, 1))
    ax_vl.set_title('Value Loss')
    ax_vl.plot(model.v_loss_hist)
    
    ax_kl = plt.subplot2grid((3, 3), (2, 2))
    ax_kl.set_title('KL Divergence')
    ax_kl.plot(model.kl_hist)

    fig.canvas.draw()

fig.savefig('ppo_result_'+str(datetime.datetime.now())+'.png')

