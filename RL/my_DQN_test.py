import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000

# 增加，减少
N_ACTIONS = 2

# 先只调节闪烁次数。
# 闪烁时间和间隔，再议。
N_STATES = 3

# begin_attention = 1

def getBrain():
    # 实际情况，这里就是准确率
    attention = random.randint(0, 100)
    if attention > 80:
        return 1
    elif attention > 40:
        return 0.5
    elif attention > 10:
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
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net = Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        actions_value = self.eval_net(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0]
        # print(actions_value)
        return action


dqn = DQN()
state = torch.load('model.pth')
dqn.eval_net.load_state_dict(state)
action = ['减少', '增加']


jia = 0
jian = 0
old = 0.3
for i in range(3, 21):
    for j in range(80, 150, 10):
        new_ = getBrain()
        a = dqn.choose_action([i, old,new_])
        old = new_
        if a == 0:
            jian += 1
        else:
            jia += 1
        print('( 闪烁次数：', i, '闪烁时间：', j,') action：', action[a])

print(jia, jian)
