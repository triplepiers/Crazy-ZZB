# Desc: 实际提交的版本 => test 给的 epoch 实在太小了，练不出来

# 导入相关包
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn.functional as F 

from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.QNetwork import QNetwork
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本

def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []

    # -----------------请实现你的算法代码--------------------------------------
    N = maze.maze_size-1
    cur_pos = maze.sense_robot()

    def opposite(direct):
        if direct == 'u':
            return 'd'
        elif direct == 'd':
            return 'u'
        elif direct == 'l':
            return 'r'
        elif direct == 'r':
            return 'l'

    def DFS(cur_pos):
        i, j = cur_pos
        # go out
        if cur_pos == (N, N):
            return True
        else:
            options = maze.can_move_actions(cur_pos)

            # no going BACK
            if len(path) > 0:
                options.remove(opposite(path[-1])) 
            
            # no where to go
            if len(options) == 0:
                path.pop()
                return False
            else:
                for direction in options:
                    path.append(direction)
                    if direction == 'u':
                        if DFS((i-1, j)):
                            return True
                    elif direction == 'd':
                        if DFS((i+1, j)):
                            return True
                    elif direction == 'l':
                        if DFS((i, j-1)):
                            return True
                    elif direction == 'r':
                        if DFS((i, j+1)):
                            return True
                # no where to go
                path.pop()
                return False

    DFS(cur_pos)
    # -----------------------------------------------------------------------
    return path

# additional: 修改 GQNetwork 中 NN 的隐藏层配置

class Robot(QRobot):

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)

        self.step = 1
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        m_size = maze.maze_size
        # update reward (Minimize)
        maze.set_reward({
            "hit_wall": 10.,
            "destination": -5. * m_size**2,
            "default": 1.,
        })
        self.maze = maze
        self.maze_size = m_size

        # defaults
        self.updateInterval = m_size**2 - 1

        # init memo
        self.memory = ReplayDataSet(max_size=max(
            1e4, m_size**2 * 3
        ))
        self.memory.build_full_view(maze)

        # init NNs
        self.batch_size         = len(self.memory.Experience)
        self.init_learning_rate = 0.001
        target_nn, eval_nn = None, None
        self._build_nn()
        
        self.train_myself()
        return
    
    def train_myself(self):
        batch_size = len(self.memory) 
        m_size     = self.maze.maze_size
        dst_reward = self.maze.reward['destination']

        while True:
            loss = self.learn(batch_size)
            self.reset()
            for _ in range(m_size ** 2 - 1):
                a, r = self.test_update()
                if r == dst_reward:
                    return 

    def _build_nn(self):
        seed = 0
        random.seed(seed)

        nn_config = {
            'state_size':  2,
            'action_size': 4,
            'seed':        seed,
#             'mine':        True
        }

        # init NN
        self.eval_nn   = QNetwork(**nn_config).to(self.device)
        self.target_nn = QNetwork(**nn_config).to(self.device)

        # init optimizer(Adam)
        self.optimizer = optim.Adam(
            self.eval_nn.parameters(),
            lr=self.init_learning_rate
        )

        return
    
    def _update_target_nn(self):
        self.target_nn.load_state_dict(self.eval_nn.state_dict())
        return

    def _get_reward(self, action):
        return self.maze.move_robot(action)

    def _get_action_idx(self, action):
        return self.valid_action.index(action)
    
    def _choose_action(self, state):
        state = torch.from_numpy(
            np.array(self.sense_state(), dtype=np.int16)
        ).float().to(self.device)

        if random.random() < self.epsilon:
            return random.choice(self.valid_action)
        else:
            self.eval_nn.eval()
            with torch.no_grad():
                q_next = self.eval_nn(state).cpu().data.numpy() 
            self.eval_nn.train()
            return self.valid_action[np.argmin(q_next).item()]

    def learn(self, batch_size):
        if len(self.memory.Experience) < batch_size:
            return

        state, action_idx, reward, next_state, is_terminal = self.memory.random_sample(batch_size)
        
        state       = torch.from_numpy(state).float().to(self.device)
        action_idx  = torch.from_numpy(action_idx).long().to(self.device)
        reward      = torch.from_numpy(reward).float().to(self.device)
        next_state  = torch.from_numpy(next_state).float().to(self.device)
        is_terminal = torch.from_numpy(is_terminal).int().to(self.device)

        self.eval_nn.train()
        self.target_nn.eval()

        q_next = self.target_nn(next_state).detach().min(1)[0].unsqueeze(1)
        q_cur  = reward + self.gamma * q_next * (
            torch.ones_like(is_terminal) - is_terminal
        )

        self.optimizer.zero_grad()
        q_pred = self.eval_nn(state).gather(dim=1, index=action_idx)

        loss = F.mse_loss(q_pred, q_cur)
        loss_val = loss.item()
        loss.backward()
        self.optimizer.step()

        self._update_target_nn()

        return loss_val

    def train_update(self):
        """
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """

        # -----------------请实现你的算法代码--------------------------------------
        state  = self.sense_state()
        action = self._choose_action(state)
        reward = self._get_reward(action)
        # -----------------------------------------------------------------------

        return action, reward

    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return : action, reward 如："u", -1
        """
        # -----------------请实现你的算法代码--------------------------------------
        state = torch.from_numpy(
            np.array(self.sense_state(), dtype=np.int16)
        ).float().to(self.device)
        
        self.eval_nn.eval()
        with torch.no_grad():
            q_vals = self.eval_nn(state).cpu().data.numpy()

        action = self.valid_action[np.argmin(q_vals).item()]
        reward = self._get_reward(action)
        # -----------------------------------------------------------------------
        return action, reward
