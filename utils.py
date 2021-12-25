import copy
from collections import namedtuple
import random
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

def load_env(worker_id, base_port, file="../Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe",
             grap=True, train=True):
    """
    load Unity Environment
    :param worker_id: ID env
    :param base_port: communications port with unity agent
    """

    env = UnityEnvironment(file_name=file, worker_id=worker_id, base_port=base_port, no_graphics=grap)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size
    # reset the environment
    env_info = env.reset(train_mode=train)[brain_name]
    # number agents
    n_agents = len(env_info.agents)
    state = env_info.vector_observations[0]
    state_size = len(state)
    return env , brain_name, brain, action_size, env_info, state, state_size, n_agents

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class GaussianNoise:
    """Gaussian Noise.
    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        action_dim: int,
        min_sigma: float = 1.0,
        max_sigma: float = 1.0,
        decay_period: int = 1000000,
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.DEVICE = device

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def plot_scores(scores , algo, num_episodes):
    """
    plot scores and save to disk
    :param scores: list of scores during training
    :param algo: type algorithm
    :return:
    """
    # plot the scores
    text = ""
    if algo == "1":
        text= "DDPG Agent"
    elif algo == "3":
        text = "Dueling DQN Agent with Priority Buffer"
    elif algo == "4":
        text = "Categorical DQN Agent No prioritary buffer"
    elif algo == "2":
        text = "Dueling DQN Agent"
    elif algo == "5":
        text = "Dueling Noisy DQN Agent with Priority Buffer"
    elif algo == "6":
        text = "DQN n-Steps Agent"
    elif algo == "7":
        text = "DQN Rainbow Agent"
    elif algo == "8":
        text = "Dueling Noisy DQN Agent No PER"
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'Algo {text} Number episodes {num_episodes}')
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'./images/scores_{algo}.jpg')
    return

def plot_losses(losses , algo, num_episodes):
    """
    plot scores and save to disk
    :param scores: list of scores during training
    :param algo: type algorithm
    :return:
    """
    # plot the scores
    text = ""
    if algo == "1":
        text= "DDPG Agent"
    elif algo == "3":
        text = "Dueling DQN Agent with Priority Buffer"
    elif algo == "4":
        text = "Categorical DQN Agent No prioritary buffer"
    elif algo == "2":
        text = "Dueling DQN Agent"
    elif algo == "5":
        text = "Dueling Noisy DQN Agent with Priority Buffer"
    elif algo == "6":
        text = "DQN n-Steps Agent"
    elif algo == "7":
        text = "DQN Rainbow Agent"
    elif algo == "8":
        text = "Dueling Noisy DQN Agent No PER"
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'Algo {text} Number episodes {num_episodes}')
    plt.plot(np.arange(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('Episode #')
    plt.savefig(f'./images/losses_{algo}.jpg')
    return