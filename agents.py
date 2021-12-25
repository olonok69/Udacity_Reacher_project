import torch

from networks import *
from utils import *
import torch.optim as optim
import os
import random
from torch.autograd import Variable


class Agent_DDPG():
    def __init__(self,
                 device,
                 state_size,
                 n_agents,
                 action_size,
                 random_seed,
                 buffer_size,
                 batch_size,
                 gamma,
                 TAU,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 algo,
                 checkpoint_folder = './checkpoints/',
                 train=True):
        """
        DDPG Agent implementation
        :param device: CPU or CUDA
        :type device: string
        :param state_size: env number of states (33)
        :type state_size: int
        :param n_agents: number of arms this agent (20 or 1)
        :type n_agents: int
        :param action_size: number of actions of this env (4)
        :type action_size: int
        :param random_seed: ramdom seed to repeat experiment and initialize weigths
        :type random_seed:  int
        :param buffer_size: size of buffer
        :type buffer_size: int
        :param batch_size: size of batch
        :type batch_size: int
        :param gamma: discount reward parameter
        :type gamma: float
        :param TAU: discount parameter soft update models
        :type TAU: float
        :param lr_actor: learning rate actor
        :type lr_actor: float
        :param lr_critic: learning rate critic
        :type lr_critic: float
        :param weight_decay: decay parameter learning rate
        :type weight_decay: float
        :param algo: type algorithm
        :type algo: string
        :param checkpoint_folder: folder for final models
        :type checkpoint_folder: strings
        :param train: mode train NN
        :type train: bool
        """
        self.DEVICE = device
        self.train = train
        self.algo = algo

        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # Actor Network (w/ Target Network)
        self.actor_local = Actor_ddpg(state_size, action_size, random_seed).to(self.DEVICE)
        self.actor_target = Actor_ddpg(state_size, action_size, random_seed).to(self.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic_ddpg(state_size, action_size, random_seed).to(self.DEVICE)
        self.critic_target = Critic_ddpg(state_size, action_size, random_seed).to(self.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC,
                                           weight_decay=self.WEIGHT_DECAY)

        if os.path.isfile(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.algo}.pth') and self.train==False:
            self.actor_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth'))
            self.actor_target.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth'))

            self.critic_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.algo}.pth'))
            self.critic_target.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.algo}.pth'))

        # Noise process
        self.noise = OUNoise((n_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(device, action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        # losses
        self.losses_actor = []
        self.losses_critic = []

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.n_agents):
            self.memory.add(state[i ,:], action[i ,:], reward[i], next_state[i ,:], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, episodes, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)# we clip the action values to the constrains required on the problem statement

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # record loss
        self.losses_critic.append(critic_loss.item())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # record loss
        self.losses_actor.append(actor_loss.item())
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        self.actor_loss = actor_loss.data
        self.critic_loss = critic_loss.data

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)

    def checkpoint(self, algo, folder):
        torch.save(self.actor_local.state_dict(), folder + f'checkpoint_actor_{algo}.pth')
        torch.save(self.critic_local.state_dict(), folder + f'checkpoint_critic_{algo}.pth')

class Agent_TD3():
    def __init__(self,
                 device,
                 state_size,
                 n_agents,
                 action_size,
                 random_seed,
                 buffer_size,
                 batch_size,
                 gamma,
                 TAU,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 algo,
                 exploration_noise = 0.1,
                 target_policy_noise = 0.2,
                 target_policy_noise_clip = 0.5,
                 action_noise= False,
                 initial_random_steps = int(1e4),
                 checkpoint_folder = './checkpoints/',
                 train=True):
        """
        TD3 Agent implementation
        :param device: CPU or CUDA
        :type device: string
        :param state_size: env number of states (33)
        :type state_size: int
        :param n_agents: number of arms this agent (20 or 1)
        :type n_agents: int
        :param action_size: number of actions of this env (4)
        :type action_size: int
        :param random_seed: ramdom seed to repeat experiment and initialize weigths
        :type random_seed:  int
        :param buffer_size: size of buffer
        :type buffer_size: int
        :param batch_size: size of batch
        :type batch_size: int
        :param gamma: discount reward parameter
        :type gamma: float
        :param TAU: discount parameter soft update models
        :type TAU: float
        :param lr_actor: learning rate actor
        :type lr_actor: float
        :param lr_critic: learning rate critic
        :type lr_critic: float
        :param weight_decay: decay parameter learning rate
        :type weight_decay: float
        :param algo: type algorithm
        :type algo: string
        :param exploration_noise:  gaussian noise for policy
        :type exploration_noise: float
        :param target_policy_noise: gaussian noise for target policy
        :type target_policy_noise: float
        :param target_policy_noise_clip: clip target gaussian noise
        :type target_policy_noise_clip: float
        :param action_noise: use noise on target next_action
        :type action_noise: bool
        :param initial_random_steps:  initial random action steps
        :type initial_random_steps: int
        :param checkpoint_folder: folder for final models
        :type checkpoint_folder: strings
        :param train: mode train NN
        :type train: bool
        """

        # total steps count
        self.total_step = 0
        # update step for actor
        self.update_step = 2

        self.DEVICE = device
        self.train = train
        self.algo = algo

        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # noise
        # exploration
        self.exploration_noise = GaussianNoise(
            action_size, exploration_noise, exploration_noise
        )
        # Npise actions
        self.target_policy_noise = GaussianNoise(
            action_size, target_policy_noise, target_policy_noise
        )
        self.target_policy_noise_clip = target_policy_noise_clip
        # introduce noise / not introduce Noise
        self.action_noise = action_noise

        # Actor Network (w/ Target Network)
        self.actor_local = Actor_TD3(state_size, action_size).to(self.DEVICE)
        self.actor_target = Actor_TD3(state_size, action_size).to(self.DEVICE)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network 1 (w/ Target Network)
        self.critic_local_1 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_1 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_1.load_state_dict(self.critic_local_1.state_dict())

        # Critic Network 2 (w/ Target Network)
        self.critic_local_2 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_2 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_2.load_state_dict(self.critic_local_2.state_dict())

        self.initial_random_steps =initial_random_steps
        # concat critic parameters to use one optim
        critic_parameters = list(self.critic_local_1.parameters()) + list(
            self.critic_local_2.parameters()
        )
        self.critic_optimizer = optim.Adam(critic_parameters, lr=self.LR_CRITIC)

        if os.path.isfile(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_1_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_2_{self.algo}.pth') and self.train==False:
            # load models from files
            self.actor_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth'))
            self.critic_local_1.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_1_{self.algo}.pth'))
            self.critic_local_2.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_2_{self.algo}.pth'))


        # Replay memory
        self.memory = ReplayBuffer(device, action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        # losses
        self.losses_actor = []
        self.losses_critic = []

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.n_agents) :
            self.memory.add(state[i ,:], action[i ,:], reward[i], next_state[i ,:], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        if self.train == True :
            self.total_step = episode + 1
        if self.total_step < self.initial_random_steps and self.train ==True:
            pass
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.exploration_noise.sample()
        return np.clip(action, -1, 1)# we clip the action values to the constrains required on the problem statement

    def reset(self):
        pass


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.train = True
        states, actions, rewards, next_states, dones = experiences

        # get actions with noise
        if self.action_noise:
            noise = torch.FloatTensor(self.target_policy_noise.sample()).to(self.DEVICE)
            clipped_noise = torch.clamp(
                noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
            )
            next_actions = (self.actor_target(next_states) + clipped_noise).clamp(
                -1.0, 1.0
            )
        else:
            next_actions = self.actor_target(next_states).clamp(
            -1.0, 1.0
            )
        # min (Q_1', Q_2')
        #print(next_actions)
        next_values1 = self.critic_target_1(next_states, next_actions)
        next_values2 = self.critic_target_2(next_states, next_actions)
        #print(next_values1, next_values2)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        Q_targets = rewards + (self.GAMMA * next_values * (1 - dones))
        Q_targets = Q_targets.detach()

        # critic loss
        values1 = self.critic_local_1(states, actions)
        values2 = self.critic_local_2(states, actions)
        critic1_loss = F.mse_loss(values1, Q_targets)
        critic2_loss = F.mse_loss(values2, Q_targets)

        # train critic
        critic_loss = critic1_loss + critic2_loss
        # record loss
        self.losses_critic.append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        if self.total_step % self.update_step == 0:
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, actions_pred).mean()

            # record loss
            self.losses_actor.append(actor_loss.item())

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic_local_1, self.critic_target_1)
            self.soft_update(self.actor_local, self.actor_target)
            self.soft_update(self.critic_local_2, self.critic_target_2)
        else:
            self.losses_actor.append(torch.zeros(1))

        # self.actor_loss = actor_loss.data
        # self.critic_loss = critic_loss.data
        return


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)

    def checkpoint(self,algo, folder):
        torch.save(self.actor_local.state_dict(), folder + f'checkpoint_actor_{algo}.pth')
        torch.save(self.critic_local_1.state_dict(), folder + f'checkpoint_critic_1_{algo}.pth')
        torch.save(self.critic_local_2.state_dict(), folder + f'checkpoint_critic_2_{algo}.pth')



class Agent_TD3_4():
    def __init__(self,
                 device,
                 state_size,
                 n_agents,
                 action_size,
                 random_seed,
                 buffer_size,
                 batch_size,
                 gamma,
                 TAU,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 algo,
                 exploration_noise = 0.1,
                 target_policy_noise = 0.2,
                 target_policy_noise_clip = 0.5,
                 action_noise= False,
                 initial_random_steps = int(1e4),
                 checkpoint_folder = './checkpoints/',
                 train=True):
        """
        TD3 Agent implementation
        :param device: CPU or CUDA
        :type device: string
        :param state_size: env number of states (33)
        :type state_size: int
        :param n_agents: number of arms this agent (20 or 1)
        :type n_agents: int
        :param action_size: number of actions of this env (4)
        :type action_size: int
        :param random_seed: ramdom seed to repeat experiment and initialize weigths
        :type random_seed:  int
        :param buffer_size: size of buffer
        :type buffer_size: int
        :param batch_size: size of batch
        :type batch_size: int
        :param gamma: discount reward parameter
        :type gamma: float
        :param TAU: discount parameter soft update models
        :type TAU: float
        :param lr_actor: learning rate actor
        :type lr_actor: float
        :param lr_critic: learning rate critic
        :type lr_critic: float
        :param weight_decay: decay parameter learning rate
        :type weight_decay: float
        :param algo: type algorithm
        :type algo: string
        :param exploration_noise:  gaussian noise for policy
        :type exploration_noise: float
        :param target_policy_noise: gaussian noise for target policy
        :type target_policy_noise: float
        :param target_policy_noise_clip: clip target gaussian noise
        :type target_policy_noise_clip: float
        :param action_noise: use noise on target next_action
        :type action_noise: bool
        :param initial_random_steps:  initial random action steps
        :type initial_random_steps: int
        :param checkpoint_folder: folder for final models
        :type checkpoint_folder: strings
        :param train: mode train NN
        :type train: bool
        """

        # total steps count
        self.total_step = 0
        # update step for actor
        self.update_step = 2

        self.DEVICE = device
        self.train = train
        self.algo = algo

        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # noise
        # exploration
        self.exploration_noise = GaussianNoise(
            action_size, exploration_noise, exploration_noise
        )
        # Npise actions
        self.target_policy_noise = GaussianNoise(
            action_size, target_policy_noise, target_policy_noise
        )
        self.target_policy_noise_clip = target_policy_noise_clip
        # introduce noise / not introduce Noise
        self.action_noise = action_noise

        # Actor Network (w/ Target Network)
        self.actor_local = Actor_TD3(state_size, action_size).to(self.DEVICE)
        self.actor_target = Actor_TD3(state_size, action_size).to(self.DEVICE)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network 1 (w/ Target Network)
        self.critic_local_1 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_1 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_1.load_state_dict(self.critic_local_1.state_dict())

        # Critic Network 2 (w/ Target Network)
        self.critic_local_2 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_2 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_2.load_state_dict(self.critic_local_2.state_dict())

        # Critic Network 3 (w/ Target Network)
        self.critic_local_3 = Critic_TD3(state_size + action_size).to(self.DEVICE)
        self.critic_target_3 = Critic_TD3(state_size + action_size).to(self.DEVICE)
        self.critic_target_3.load_state_dict(self.critic_local_3.state_dict())

        # Critic Network 4 (w/ Target Network)
        self.critic_local_4 = Critic_TD3(state_size + action_size).to(self.DEVICE)
        self.critic_target_4 = Critic_TD3(state_size + action_size).to(self.DEVICE)
        self.critic_target_4.load_state_dict(self.critic_local_4.state_dict())

        self.initial_random_steps =initial_random_steps
        # concat critic parameters to use one optim
        critic_parameters = list(self.critic_local_1.parameters()) + list(
            self.critic_local_2.parameters()) + list(self.critic_local_3.parameters()) +  \
                            list(self.critic_local_4.parameters())

        self.critic_optimizer = optim.Adam(critic_parameters, lr=self.LR_CRITIC)

        if os.path.isfile(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_1_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_2_{self.algo}.pth') and self.train==False:
            # load models from files
            self.actor_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth'))
            self.critic_local_1.load_state_dict(
                torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_1_{self.algo}.pth'))
            self.critic_local_2.load_state_dict(
                torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_2_{self.algo}.pth'))
            self.critic_local_3.load_state_dict(
                torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_3_{self.algo}.pth'))
            self.critic_local_4.load_state_dict(
                torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_4_{self.algo}.pth'))


        # Replay memory
        self.memory = ReplayBuffer(device, action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        # losses
        self.losses_actor = []
        self.losses_critic = []

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.n_agents) :
            self.memory.add(state[i ,:], action[i ,:], reward[i], next_state[i ,:], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        if self.train == True :
            self.total_step = episode + 1
        if self.total_step < self.initial_random_steps and self.train ==True:
            pass
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.exploration_noise.sample()
        return np.clip(action, -1, 1)# we clip the action values to the constrains required on the problem statement

    def reset(self):
        pass


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.train = True
        states, actions, rewards, next_states, dones = experiences

        # get actions with noise
        if self.action_noise:
            noise = torch.FloatTensor(self.target_policy_noise.sample()).to(self.DEVICE)
            clipped_noise = torch.clamp(
                noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
            )
            next_actions = (self.actor_target(next_states) + clipped_noise).clamp(
                -1.0, 1.0
            )
        else:
            next_actions = self.actor_target(next_states).clamp(
            -1.0, 1.0
            )
        # min (Q_1', Q_2')
        #print(next_actions)
        next_values1 = self.critic_target_1(next_states, next_actions)
        next_values2 = self.critic_target_2(next_states, next_actions)
        next_values3 = self.critic_target_3(next_states, next_actions)
        next_values4 = self.critic_target_4(next_states, next_actions)
        #print(next_values1, next_values2)
        next_values_1 = torch.min(next_values1, next_values2)
        next_values_2 = torch.min(next_values3, next_values4)
        next_values = torch.min(next_values_1, next_values_2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        Q_targets = rewards + (self.GAMMA * next_values * (1 - dones))
        Q_targets = Q_targets.detach()

        # critic loss
        values1 = self.critic_local_1(states, actions)
        values2 = self.critic_local_2(states, actions)
        values3 = self.critic_local_3(states, actions)
        values4 = self.critic_local_4(states, actions)
        critic1_loss = F.mse_loss(values1, Q_targets)
        critic2_loss = F.mse_loss(values2, Q_targets)
        critic3_loss = F.mse_loss(values3, Q_targets)
        critic4_loss = F.mse_loss(values4, Q_targets)
        # train critic
        critic_loss = critic1_loss + critic2_loss + critic3_loss + critic4_loss
        # record loss
        self.losses_critic.append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        if self.total_step % self.update_step == 0:
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, actions_pred).mean()

            # record loss
            self.losses_actor.append(actor_loss.item())

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic_local_1, self.critic_target_1)
            self.soft_update(self.actor_local, self.actor_target)
            self.soft_update(self.critic_local_2, self.critic_target_2)
            self.soft_update(self.critic_local_3, self.critic_target_3)
            self.soft_update(self.critic_local_4, self.critic_target_4)
        else:
            self.losses_actor.append(torch.zeros(1))

        # self.actor_loss = actor_loss.data
        # self.critic_loss = critic_loss.data
        return


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)

    def checkpoint(self,algo, folder):
        torch.save(self.actor_local.state_dict(), folder + f'checkpoint_actor_{algo}.pth')
        torch.save(self.critic_local_1.state_dict(), folder + f'checkpoint_critic_1_{algo}.pth')
        torch.save(self.critic_local_2.state_dict(), folder + f'checkpoint_critic_2_{algo}.pth')
        torch.save(self.critic_local_3.state_dict(), folder + f'checkpoint_critic_3_{algo}.pth')
        torch.save(self.critic_local_4.state_dict(), folder + f'checkpoint_critic_4_{algo}.pth')

class Agent_A2C():
    def __init__(self,
                 device,
                 state_size, n_agents, action_size, random_seed,
                 gamma, TAU, lr_actor, lr_critic, weight_decay,
                 entropy_weight,algo,
                 checkpoint_folder = './models/', train=True):
        """
        A2C Agent Implementation

        """
        self.DEVICE = device
        self.train = train
        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()
        # algo number
        self.algo =algo
        #rate of weighting entropy into the loss function
        self.entropy_weight = entropy_weight

        # total steps count
        self.total_step = 0
        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters

        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # Actor Network
        self.actor_local = Actor_a2c(state_size, action_size, random_seed).to(self.DEVICE)
        # initialize with its own Learning Rate
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network
        self.critic_local = Critic_a2c(state_size, random_seed).to(self.DEVICE)
        # initialize with its own Learning Rate
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC,
                                           weight_decay=self.WEIGHT_DECAY)

        # is train is  False , load trained model from folder
        if os.path.isfile(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.algo}.pth') and self.train==False:
            self.actor_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth'))
            self.critic_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic.pth_{self.algo}'))


        # losses
        self.losses_actor = []
        self.losses_critic = []
    def reset(self):
        pass
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        if self.train:
            self.transition.extend([next_state, reward, done])

        self.learn()

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()

        with torch.no_grad():
            action, dist  = self.actor_local(state)
        self.actor_local.train()
        # if train take the action if not mean from distribution
        selected_action = dist.mean if not(self.train) else action

        if self.train:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]
        # we clip the action values to the constrains required on the problem statement
        selected_action = selected_action.cpu().detach().numpy()

        return np.clip(selected_action, -1, 1)



    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        #states, actions, rewards, next_states, dones = experiences
        self.train=True
        state, log_prob, next_state, reward, dones = self.transition
        # ---------------------------- update critic ---------------------------- #
        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        #mask = 1 - done
        mask=[]
        [mask.append(1 - done) for done in dones]
        next_state = torch.FloatTensor(next_state).to(self.DEVICE)
        mask = torch.from_numpy(np.array(mask)).to(self.DEVICE)
        reward = torch.from_numpy(np.array(reward)).to(self.DEVICE)


        pred_value = self.critic_local(state)
        pred_value2 = torch.clone(pred_value)
        nnext_state = self.critic_local(next_state)

        #targ_value =  reward + (self.GAMMA * nnext_state * mask).mean(dim=1)
        targ_value = (nnext_state * self.GAMMA * mask).mean(dim=1) + reward
        targ_value = targ_value.resize(self.n_agents, 1)
        targ_value2 =torch.clone(targ_value)
        value_loss = F.smooth_l1_loss(pred_value, targ_value)
        #print(value_loss)
        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        # Compute actor loss
        pred_value2 = pred_value2.resize(self.n_agents)
        advantage = (targ_value2 - pred_value2).detach() # not backpropagated
        #print(log_prob)
        policy_loss = Variable((-advantage * log_prob), requires_grad=True)
        policy_loss = Variable(policy_loss + (self.entropy_weight * -log_prob), requires_grad=True)
        policy_loss2 = Variable(policy_loss.mean(), requires_grad=True)
        # policy_loss = -advantage * log_prob
        # policy_loss += self.entropy_weight * -log_prob  # entropy maximization

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss2.backward()
        self.actor_optimizer.step()

        self.losses_actor.append(policy_loss2.item())
        self.losses_critic.append(value_loss.item())



    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)

    def checkpoint(self, algo, folder):
        torch.save(self.actor_local.state_dict(), folder + f'checkpoint_actor_{algo}.pth')
        torch.save(self.critic_local.state_dict(), folder + f'checkpoint_critic_{algo}.pth')

def agent_train(env,brain_name, agent, n_agents ,algo):
    scores = []
    loss_actor = []
    loss_critic = []
    scores_window = deque(maxlen=100)
    n_episodes = 1000

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations
        agent.reset()  # reset the agent noise
        score = np.zeros(n_agents)

        while True:
            actions = agent.act(states, episode)

            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            agent.step(states, actions, rewards, next_states, dones)

            score += rewards  # update the score

            states = next_states  # roll over the state to next time step

            if np.any(dones):  # exit loop if episode finished
                break

        agent.checkpoint(algo, "./checkpoints/")

        scores.append(np.mean(score))

        loss_actor.append(np.mean(agent.losses_actor))
        loss_critic.append(np.mean(agent.losses_critic))
        scores_window.append(np.mean(score))

        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f} \tLoss critic: \t{:.2f} '
              '\tloss Actor: \t{:.2f}'.format(episode, np.mean(score), np.mean(scores_window), np.mean(loss_critic)
                                                    ,np.mean(loss_actor)), end="")

        if np.mean(scores_window) >= 35.0:
            agent.checkpoint(algo, "./models/")
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            break


    plot_scores(scores,algo ,n_episodes)
    plot_losses(loss_actor, algo, n_episodes)
    return