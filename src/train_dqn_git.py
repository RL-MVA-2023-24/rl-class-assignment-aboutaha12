from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV

import random
import torch
import torch.nn as nn
from copy import deepcopy

import numpy as np
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity  # capacity of the buffer
        self.data = []
        self.index = 0  # index of the next cell to be filled
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(
            map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch)))
        )

    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class ProjectAgent:
    def model_mlp(self, state_dim, n_action, nb_neurons):
        return torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action),
        ).to(device)

    def __init__(self):
        config = {
            "nb_actions": env.action_space.n,
            "nb_neurons": 256,
        }
        self.nb_actions = config["nb_actions"]
        self.neurons = config["nb_neurons"]
        self.gamma = config["gamma"] if "gamma" in config.keys() else 0.95
        self.batch_size = config["batch_size"] if "batch_size" in config.keys() else 100
        buffer_size = (
            config["buffer_size"] if "buffer_size" in config.keys() else int(1e5)
        )
        self.model = self.model_mlp(
            env.observation_space.shape[0], self.nb_actions, self.neurons
        )
        self.memory = ReplayBuffer(buffer_size, device)
        self.epsilon_max = (
            config["epsilon_max"] if "epsilon_max" in config.keys() else 1.0
        )
        self.epsilon_min = (
            config["epsilon_min"] if "epsilon_min" in config.keys() else 0.01
        )
        self.epsilon_stop = (
            config["epsilon_decay_period"]
            if "epsilon_decay_period" in config.keys()
            else 1000
        )
        self.epsilon_delay = (
            config["epsilon_delay_decay"]
            if "epsilon_delay_decay" in config.keys()
            else 20
        )
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = (
            config["criterion"] if "criterion" in config.keys() else torch.nn.MSELoss()
        )
        lr = config["learning_rate"] if "learning_rate" in config.keys() else 0.001
        self.optimizer = (
            config["optimizer"]
            if "optimizer" in config.keys()
            else torch.optim.Adam(self.model.parameters(), lr=lr)
        )
        self.nb_gradient_steps = (
            config["gradient_steps"] if "gradient_steps" in config.keys() else 1
        )
        self.update_target_strategy = (
            config["update_target_strategy"]
            if "update_target_strategy" in config.keys()
            else "replace"
        )
        self.update_target_freq = (
            config["update_target_freq"]
            if "update_target_freq" in config.keys()
            else 20
        )
        self.update_target_tau = (
            config["update_target_tau"]
            if "update_target_tau" in config.keys()
            else 0.005
        )
        self.monitoring_nb_trials = (
            config["monitoring_nb_trials"]
            if "monitoring_nb_trials" in config.keys()
            else 0
        )

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        random = 0
        episode_length = 0
        previous_val = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                random += 1
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == "replace":
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == "ema":
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = (
                        tau * model_state_dict + (1 - tau) * target_state_dict
                    )
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            episode_length += 1
            if done or trunc:
                episode += 1
                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                episode_return.append(episode_cum_reward)
                print(
                    "Episode ",
                    "{:2d}".format(episode),
                    ", epsilon ",
                    "{:6.2f}".format(epsilon),
                    ", ep return ",
                    "{:.2e}}".format(episode_cum_reward),
                    ", rand ",
                    "{:2d}".format(random),
                    "/",
                    "{:2d}".format(episode_length),
                    " validation score ",
                    "{:.2e}".format(validation_score),
                    sep="",
                )
                # add an early stopping ?
                if validation_score > previous_val:
                    print(
                        f"Better model: Validation Score {validation_score:.2e}",
                    )

                    previous_val = validation_score
                    self.best_model = deepcopy(self.model).to(device)
                    path = os.getcwd()
                    self.save(path)
                state, _ = env.reset()
                episode_cum_reward = 0
                episode_length = 0
                random = 0
            else:
                state = next_state
        return episode_return

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        # Act greedily if no random
        return greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(
            torch.load(
                "src/best_model_Taha.pt",
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()
