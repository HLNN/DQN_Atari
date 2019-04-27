import sys
import argparse
import random
import math
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import gym
from gym import wrappers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
# from tensorboardX import SummaryWriter

import pdb
from skimage.transform import resize
from skimage.color import rgb2gray

import cv2
import time
import os


RENDER = False
EVN = 'SpaceInvaders-v0'
EPISODE = 1000000
BATCH_SIZE = 128
GAMMA = 0.9
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY = 1000000
MEMORY_BUFFER = 10000
MEMORY_BURN_LIMIT = 2000
TARGET_UPDATE = 10
LR = 0.0001


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9 * 9 * 32, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 9 * 9 * 32)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self):
        self.policy_net, self.target_net = CNN(), CNN()

        if use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        self.memory = ReplayMemory(MEMORY_BUFFER)

        self.learn_step_counter = 0
        self.batch_size = BATCH_SIZE
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        if use_cuda:
            next_state_values = next_state_values.cuda()
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class Agent:
    def __init__(self):
        self.env = gym.make(EVN)
        self.render = RENDER
        self.dpn = DQN()

        self.steps = 0

        self.batch_size = BATCH_SIZE
        self.epsilon_start = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA
        self.memory_burn_limit = MEMORY_BURN_LIMIT

    def preprocessing(self, observation):
        state = rgb2gray(observation)
        state = state[30:-14, 2:-16]
        state = resize(state, (84, 84))
        return state

    def select_action(self, state, train):
        if train:
            self.steps += 1

        if train:
            epsilon = self.epsilon_start - self.steps * (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        else:
            epsilon = 0.05

        state = FloatTensor(state)
        if random.random() > epsilon:
            actions_value = self.dpn.policy_net.forward(state)
            if use_cuda:
                action = torch.max(actions_value, 1)[1].cpu().data.numpy()[0]
            else:
                action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action = random.randint(0, self.env.action_space.n - 1)
        return action

    def burn_memory(self):
        if os.path.exists('model.pkl'):
            print('Find existing model, starting loading...')
            self.dpn.policy_net.load_state_dict(torch.load('model.pkl'))
            self.dpn.target_net.load_state_dict(torch.load('model.pkl'))
            print('Model loaded, ready to start training now')
            return

        steps = 0
        state = np.zeros((4, 84, 84))
        next_state = np.zeros((4, 84, 84))

        state_single = self.env.reset()
        state_single = self.preprocessing(state_single)

        state[0, :, :] = state_single
        state[1, :, :] = state_single
        state[2, :, :] = state_single
        state[3, :, :] = state_single

        print('Starting to fill the memory with random policy')

        while steps < self.memory_burn_limit:
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)

            next_state_single = self.preprocessing(observation)
            next_state[0, :, :] = state[1, :, :]
            next_state[1, :, :] = state[2, :, :]
            next_state[2, :, :] = state[3, :, :]
            next_state[3, :, :] = next_state_single

            if done:
                self.dpn.memory.push(
                    FloatTensor([state]),
                    LongTensor([[action]]),
                    None,
                    FloatTensor([reward])
                )
            else:
                self.dpn.memory.push(
                    FloatTensor([state]),
                    LongTensor([[action]]),
                    FloatTensor([next_state]),
                    FloatTensor([reward])
                )

            steps += 1
            state = next_state

            if done:
                state_single = self.env.reset()
                state_single = self.preprocessing(state_single)

                state[0, :, :] = state_single
                state[1, :, :] = state_single
                state[2, :, :] = state_single
                state[3, :, :] = state_single

        print('Memory filled, ready to start training now')
        print("-" * 50)

    def optimize_model(self):
        self.dpn.learn()

    def play(self, episode, train=True):
        start_time = time.time()
        state_single = self.env.reset()
        state_single = self.preprocessing(state_single)

        state = np.zeros((4, 84, 84))
        next_state = np.zeros((4, 84, 84))

        state[0, :, :] = state_single
        state[1, :, :] = state_single
        state[2, :, :] = state_single
        state[3, :, :] = state_single

        steps = 0
        total_reward = 0

        while True:
            if self.render:
                self.env.render()
            action = self.select_action([state], train)
            observation, reward, done, info = self.env.step(action)

            next_state_single = self.preprocessing(observation)
            next_state[0, :, :] = state[1, :, :]
            next_state[1, :, :] = state[2, :, :]
            next_state[2, :, :] = state[3, :, :]
            next_state[3, :, :] = next_state_single

            total_reward += reward

            if done:
                self.dpn.memory.push(
                    FloatTensor([state]),
                    LongTensor([[action]]),
                    None,
                    FloatTensor([reward])
                )
            else:
                self.dpn.memory.push(
                    FloatTensor([state]),
                    LongTensor([[action]]),
                    FloatTensor([next_state]),
                    FloatTensor([reward])
                )

            if train:
                self.optimize_model()

            state = next_state
            steps += 1

            if done:
                end_time = time.time()
                duration = end_time - start_time
                if train:
                    print(
                        "Episode {} reward achieved {} | completed after {} steps |"
                        " Total steps = {} | Time used = {} s".format(
                            episode, total_reward, steps, self.steps, duration)
                    )
                return total_reward

    def train(self, num_episodes):
        print("Going to be training for a total of {} episodes".format(num_episodes))
        for episode in range(num_episodes):
            # print("----------- Episode {} -----------".format(e))
            self.play(episode, train=True)
            if episode % 10 == 0:
                torch.save(self.dpn.policy_net.state_dict(), 'model.pkl')
                self.dpn.target_net.load_state_dict(self.dpn.policy_net.state_dict())
                if episode % 100 == 0:
                    self.test(2)

    def test(self, num_episodes):
        reward_sum = 0
        print("-"*50)
        print("Testing for {} episodes".format(num_episodes))
        for episode in range(num_episodes):
            reward_sum += self.play(episode, train=False)
        print("Running policy after training for {} updates".format(self.steps))
        print("Avg reward achieved in {} episodes : {}".format(num_episodes, reward_sum/num_episodes))
        print("-"*50)

    def close(self):
        if self.render:
            self.env.render(close=True)
        self.env.close()


def main():
    agent = Agent()

    agent.burn_memory()
    agent.train(EPISODE)
    print('----------- Completed Training -----------')
    agent.test(num_episodes=100)
    print('----------- Completed Testing -----------')

    agent.close()


if __name__ == '__main__':
    main()
