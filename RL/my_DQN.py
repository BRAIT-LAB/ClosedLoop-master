import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000

# 增加，减少
N_ACTIONS = 2

# 2指的是闪烁次数和闪烁时间
N_STATES = 3

# 准确率
old_attention = 0.3


def getBrain():
    # 实际情况，这里就是准确率
    attention = random.randint(0, 100)
    if attention > 90:
        return 1
    elif attention > 50:
        return 0.5
    elif attention > 20:
        return 0.25
    else:
        return 0
    # 此处不当作θ/β，可以当作P300的幅值、强度、准确率之类的
    # 然后需要找文献，证明 注意力水平与P300强度有线性关系
    # =====================================================
    # 当达到某一个state后，agent的策略是选择一个最优的action
    # 但是此处用随机值训练的参数，可能不会形成此策略
    # 只形成了整体策略:P300幅度增，正反馈；减，负反馈。
    # =====================================================


def getAttention(s, a, learn_num):  # 计算注意力

    if a < 0.6:
        s[0] -= 1
    else:
        s[0] += 1

    # 不是到终点才有返回值，而是每一步都有返回值
    if s[2] > s[1]:
        reward = 1
    elif s[1] == s[2] and s[2] != 0:
        reward = 0.1
    elif s[2] == 0:
        reward = -3
    else:
        reward = -1

    # 判断是否结束
    if s[0] <= 5 or s[0] >= 15 or learn_num >= 50:
        done = True
    else:
        done = False

    return s, reward, done

class Net(nn.Module):
    # 输入四个参数的状态，输出左移还是右移
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # 随机初始化
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        # 需要时不时将eval_net参数转移给target_net，实现延迟更新
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        # 10% 随机action
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action

        return action

    def store_transition(self, s, a, r, s_):
        # 记忆库，存储学习的东西，Q表
        transition = np.hstack((s, [a, r], s_))
        # 覆盖掉老的记忆
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        # 判断要不要更新参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        # 从2000个里面选出一批32个？
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # 参数含义是state，action，reward，state'.一个state里面有四个参数
        # b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        # b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        # b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # net返回的是二列，gather是只取和动作相关的那一列
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate

        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        # 就是强化学习的核心，Q(s,a) = r + γ*maxQ(s',a')
        # 在qlearning中是使用了等号直接赋值
        # 在此处是计算均方误差，然后反向传播修改参数，使等号左侧的值尽量逼近右侧。
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(600):
    new_attention = getBrain()
    # 闪烁次数、闪烁时间、旧注意力、新注意力
    s = [10, old_attention, new_attention]

    ep_r = 0
    learn_num = 1
    # attention = begin_attention

    while True:
        # 就是预测一个y'
        a = dqn.choose_action(s)

        # 执行这个动作，返回值为移动和状态, 奖励, 是否结束, info辅助信息
        s_, r, done = getAttention(s, a, learn_num)  # 进行action后的attention水平

        dqn.store_transition(s, a, r, s_)
        ep_r += abs(r)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', ep_r)

        s = s_
        old_attention = s[2]
        s[2] = getBrain()
        learn_num += 1

        if done:
            break

torch.save(dqn.target_net.state_dict(), 'model.pth')
