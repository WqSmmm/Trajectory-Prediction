import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np
import random
import pybullet as p
import pybullet_data
import time
import gym
from gym import spaces
import os
import math
from collections import deque
import torch.optim as optim

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        """存储转换到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # 确保所有数据都是numpy数组
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        """从缓冲区采样一个批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        # 分别处理每种数据
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.float32)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回当前存储的转换数量"""
        return len(self.buffer)

# 基础网络模型
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNet, self).__init__()
        '''定义值网络
        '''
        self.linear1 = nn.Linear(state_dim, hidden_dim) # 输入层
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w) # 初始化权重
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNet(nn.Module):
    """软Q网络，用于评估状态动作对的价值"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SoftQNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化最后一层的权重和偏置
        self.net[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.net[-1].bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        """前向传播
        
        Args:
            state: 状态张量
            action: 动作张量
            
        Returns:
            Q值
        """
        x = torch.cat([state, action], dim=1)  # 拼接状态和动作
        return self.net(x)
        
        
class PolicyNet(nn.Module):
    """策略网络，用于生成连续动作"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 初始化最后一层的权重和偏置
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.mean.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        """前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            mean: 动作均值
            log_std: 动作对数标准差
        """
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        return mean, log_std
    
    def sample(self, state):
        """采样动作
        
        Args:
            state: 状态张量
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()  # 使用重参数化技巧采样
        action = torch.tanh(x)  # 使用tanh压缩到[-1,1]
        
        # 计算对数概率
        log_prob = normal.log_prob(x)
        
        # 计算tanh变换后的对数概率
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        action = action.detach().cpu().numpy()
        return action[0]

# GAIL判别器网络
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)
    
    def get_reward(self, state, action):
        '''获取GAIL奖励，对应于-log(D(s,a))，其中D是判别器输出的专家概率
        '''
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action)
        with torch.no_grad():
            reward = -torch.log(self.forward(state_tensor.unsqueeze(0), action_tensor.unsqueeze(0))).item()
        return reward

# 机械臂环境
class RobotArmEnv:
    """机器人臂轨迹规划环境"""
    def __init__(self, render=False, max_steps=200):
        # PyBullet初始化
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 加载机器人模型
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        
        # 状态参数
        self.joint_angles = np.zeros(7)  # 7个关节角度
        self.end_effector_pos = np.zeros(3)  # 末端执行器位置 (x, y, z)
        self.target_pos = np.zeros(3)  # 目标位置 (x, y, z)
        self.obstacles = []  # 障碍物列表
        
        # 环境参数
        self.render_flag = render
        self.max_steps = max_steps
        self.current_step = 0
        self.workspace_bounds = {
            'x': [-1.0, 1.0],
            'y': [-1.0, 1.0],
            'z': [0.0, 1.0]
        }
        
        # 动作和状态空间维度
        self.action_dim = 7  # 7个关节角度的变化量
        self.state_dim = 16  # 7个关节角度 + 3个末端执行器位置 + 3个目标位置 + 3个最近障碍物特征
        
        # 课程学习参数
        self.current_workspace_bounds = self.workspace_bounds.copy()
        self.current_obstacle_count = 0  # 初始无障碍物
        self.current_max_steps = max_steps
        
        # 初始化随机种子
        np.random.seed(42)
        
        # 碰撞检测参数
        self.collision_threshold = 0.1
        self.end_effector_radius = 0.05
        
        # 奖励参数
        self.distance_threshold = 0.1
        self.collision_penalty = -20.0
        self.step_penalty = -0.05
        self.goal_reward = 200.0
        self.distance_reward_scale = 20.0
        self.smoothness_reward_scale = 2.0
        self.progress_reward_scale = 10.0
        
        # 重置环境
        self.reset()
    
    def update_curriculum_params(self, workspace_bounds, obstacle_count, max_steps):
        """更新课程学习参数
        
        Args:
            workspace_bounds: 工作空间范围
            obstacle_count: 障碍物数量
            max_steps: 最大步数
        """
        # 更新工作空间范围
        self.current_workspace_bounds = workspace_bounds
        
        # 更新障碍物数量
        self.current_obstacle_count = obstacle_count
        
        # 更新最大步数
        self.current_max_steps = max_steps
        
        # 清除现有障碍物
        if self.render_flag:
            for i in range(p.getNumBodies()):
                if i != self.robot_id:  # 不删除机器人
                    p.removeBody(i)
        
        # 重新生成障碍物
        self.obstacles = self._generate_obstacles(self.current_obstacle_count)
        
        # 在新的工作空间范围内重置目标位置
        self.target_pos = np.array([
            np.random.uniform(self.current_workspace_bounds['x'][0], self.current_workspace_bounds['x'][1]),
            np.random.uniform(self.current_workspace_bounds['y'][0], self.current_workspace_bounds['y'][1]),
            np.random.uniform(self.current_workspace_bounds['z'][0], self.current_workspace_bounds['z'][1])
        ])
        
        # 在PyBullet中可视化新的目标位置
        if self.render_flag:
            visual_id = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=[0, 1, 0, 0.7]  # 绿色半透明
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=self.target_pos
            )
    
    def _get_state(self):
        """获取当前状态"""
        # 获取关节状态
        joint_states = []
        for i in range(7):
            joint_state = p.getJointState(self.robot_id, i)
            joint_states.append(joint_state[0])  # 只取角度值
        
        # 获取末端执行器位置
        end_effector_state = p.getLinkState(self.robot_id, 6)
        end_effector_pos = end_effector_state[0]
        
        # 获取最近障碍物特征
        closest_obstacle_features = self._get_closest_obstacle_features()
        
        # 组合状态
        state = np.concatenate([
            np.array(joint_states),
            np.array(end_effector_pos),
            self.target_pos,
            closest_obstacle_features
        ])
        
        return state
    
    def _get_closest_obstacle_features(self):
        """获取最近障碍物的特征"""
        if not self.obstacles:
            return np.zeros(3)  # 如果没有障碍物，返回零向量
        
        end_effector_state = p.getLinkState(self.robot_id, 6)
        end_effector_pos = np.array(end_effector_state[0])
        
        min_dist = float('inf')
        closest_obstacle_pos = None
        
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle['position'])
            dist = np.linalg.norm(end_effector_pos - obstacle_pos)
            if dist < min_dist:
                min_dist = dist
                closest_obstacle_pos = obstacle_pos
        
        if closest_obstacle_pos is None:
            return np.zeros(3)
        
        # 返回到最近障碍物的相对位置
        return closest_obstacle_pos - end_effector_pos
    
    def _check_collision(self):
        """检查是否发生碰撞"""
        if not self.obstacles:
            return False
        
        end_effector_state = p.getLinkState(self.robot_id, 6)
        end_effector_pos = np.array(end_effector_state[0])
        
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle['position'])
            dist = np.linalg.norm(end_effector_pos - obstacle_pos)
            if dist < (self.end_effector_radius + obstacle['radius']):
                return True
        
        return False
    
    def _generate_obstacles(self, num_obstacles):
        """生成随机障碍物"""
        obstacles = []
        for _ in range(num_obstacles):
            position = np.array([
                np.random.uniform(self.current_workspace_bounds['x'][0], self.current_workspace_bounds['x'][1]),
                np.random.uniform(self.current_workspace_bounds['y'][0], self.current_workspace_bounds['y'][1]),
                np.random.uniform(self.current_workspace_bounds['z'][0], self.current_workspace_bounds['z'][1])
            ])
            
            radius = np.random.uniform(0.05, 0.15)
            
            # 在PyBullet中创建可视化的障碍物
            if self.render_flag:
                visual_id = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    rgbaColor=[1, 0, 0, 0.7]  # 红色半透明
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_id,
                    basePosition=position
                )
            
            obstacles.append({
                'position': position,
                'radius': radius
            })
        
        return obstacles
    
    def reset(self):
        """重置环境"""
        # 重置关节角度
        for i in range(7):
            p.resetJointState(self.robot_id, i, np.random.uniform(-np.pi, np.pi))
        
        # 重置步数计数器
        self.current_step = 0
        
        # 设置新的目标位置
        self.target_pos = np.array([
            np.random.uniform(self.current_workspace_bounds['x'][0], self.current_workspace_bounds['x'][1]),
            np.random.uniform(self.current_workspace_bounds['y'][0], self.current_workspace_bounds['y'][1]),
            np.random.uniform(self.current_workspace_bounds['z'][0], self.current_workspace_bounds['z'][1])
        ])
        
        # 在PyBullet中可视化目标位置
        if self.render_flag:
            visual_id = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=[0, 1, 0, 0.7]  # 绿色半透明
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=self.target_pos
            )
        
        # 生成新的障碍物
        self.obstacles = self._generate_obstacles(self.current_obstacle_count)
        
        # 获取初始状态
        state = self._get_state()
        
        # 初始化前一步距离
        self.prev_distance = np.linalg.norm(state[7:10] - self.target_pos)
        
        return state
    
    def step(self, action):
        """执行一步动作"""
        # 更新步数
        self.current_step += 1
        
        # 应用动作（关节角度的增量）
        for i in range(7):
            current_angle = p.getJointState(self.robot_id, i)[0]
            target_angle = current_angle + action[i]
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_angle
            )
        
        # 模拟一步
        p.stepSimulation()
        
        # 获取新状态
        new_state = self._get_state()
        
        # 计算奖励
        reward = self._compute_reward(new_state)
        
        # 检查是否完成
        done = self._is_done(new_state)
        
        # 额外信息
        info = {
            'reached_target': self._check_target_reached(new_state),
            'collision': self._check_collision(),
            'min_obstacle_distance': self._get_min_obstacle_distance()
        }
        
        return new_state, reward, done, info
    
    def _compute_reward(self, state):
        """计算奖励"""
        # 提取末端执行器位置和目标位置
        end_effector_pos = state[7:10]
        target_pos = state[10:13]
        
        # 计算到目标的距离
        distance = np.linalg.norm(end_effector_pos - target_pos)
        
        # 基础奖励（负的距离）- 增加距离奖励的权重
        reward = -distance * self.distance_reward_scale * 1.5
        
        # 如果到达目标 - 增加目标奖励
        if distance < self.distance_threshold:
            reward += self.goal_reward * 1.5
        
        # 如果发生碰撞 - 增加碰撞惩罚
        if self._check_collision():
            reward += self.collision_penalty * 1.2
        
        # 每步的惩罚 - 减少每步惩罚以鼓励探索
        reward += self.step_penalty * 0.8
        
        # 添加进度奖励 - 奖励朝目标方向的移动
        if hasattr(self, 'prev_distance'):
            # 计算距离变化
            distance_change = self.prev_distance - distance
            # 奖励朝目标方向的移动
            progress_reward = distance_change * self.progress_reward_scale * 2.0
            reward += progress_reward
        
        # 保存当前距离用于下一步计算
        self.prev_distance = distance
        
        # 添加避障奖励
        min_obstacle_distance = self._get_min_obstacle_distance()
        if min_obstacle_distance < 0.3:  # 如果接近障碍物
            obstacle_penalty = -0.5 * (0.3 - min_obstacle_distance) * self.distance_reward_scale
            reward += obstacle_penalty
        
        return reward
    
    def _is_done(self, state):
        """检查是否结束"""
        # 提取末端执行器位置和目标位置
        end_effector_pos = state[7:10]
        target_pos = state[10:13]
        
        # 计算到目标的距离
        distance = np.linalg.norm(end_effector_pos - target_pos)
        
        # 完成条件：
        # 1. 到达目标
        # 2. 发生碰撞
        # 3. 超过最大步数
        return (distance < self.distance_threshold or 
                self._check_collision() or 
                self.current_step >= self.current_max_steps)
    
    def _check_target_reached(self, state):
        """检查是否到达目标"""
        end_effector_pos = state[7:10]
        target_pos = state[10:13]
        distance = np.linalg.norm(end_effector_pos - target_pos)
        return distance < self.distance_threshold
    
    def _get_min_obstacle_distance(self):
        """获取到最近障碍物的距离"""
        if not self.obstacles:
            return float('inf')
        
        end_effector_state = p.getLinkState(self.robot_id, 6)
        end_effector_pos = np.array(end_effector_state[0])
        
        min_dist = float('inf')
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle['position'])
            dist = np.linalg.norm(end_effector_pos - obstacle_pos) - obstacle['radius']
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def close(self):
        """关闭环境"""
        p.disconnect(self.client)

# 专家数据集
class ExpertDataset:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.states), batch_size)
        return torch.FloatTensor(self.states[indices]), torch.FloatTensor(self.actions[indices])

# 专家网络（基于GAIL）
class ExpertNetwork(nn.Module):
    """专家网络，用于模仿学习"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ExpertNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围限制在[-1,1]
        )
        
        # 初始化最后一层的权重和偏置
        self.net[-2].weight.data.uniform_(-3e-3, 3e-3)
        self.net[-2].bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        """前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            action: 预测的动作
        """
        return self.net(state)

# SAC算法
class SAC:
    def __init__(self, state_dim, action_dim, 
                 hidden_dim=256, 
                 gamma=0.99, 
                 tau=0.005, 
                 alpha=0.2, 
                 lr=3e-4, 
                 batch_size=128, 
                 buffer_size=1000000, 
                 device="cpu"):
        """初始化SAC算法"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新系数
        self.alpha = alpha  # 熵调节系数
        self.batch_size = batch_size
        self.device = device
        
        # 策略网络
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        # Q网络
        self.q1_net = SoftQNet(state_dim, action_dim, hidden_dim).to(device)
        self.q2_net = SoftQNet(state_dim, action_dim, hidden_dim).to(device)
        
        # 目标Q网络
        self.target_q1_net = SoftQNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_q2_net = SoftQNet(state_dim, action_dim, hidden_dim).to(device)
        
        # 复制参数到目标网络
        self.target_q1_net.load_state_dict(self.q1_net.state_dict())
        self.target_q2_net.load_state_dict(self.q2_net.state_dict())
        
        # 优化器
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.q1_optimizer = Adam(self.q1_net.parameters(), lr=lr)
        self.q2_optimizer = Adam(self.q2_net.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
    def select_action(self, state):
        """选择动作"""
        return self.policy_net.get_action(state)
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update(self, expert_network=None, update_policy=True):
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从经验回放中采样
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # 如果使用专家网络，计算GAIL奖励
        if expert_network is not None:
            gail_rewards = []
            for i in range(self.batch_size):
                gail_reward = expert_network.get_reward(state_batch[i].cpu().numpy(), action_batch[i].cpu().numpy())
                gail_rewards.append(gail_reward)
            gail_rewards = torch.FloatTensor(gail_rewards).unsqueeze(1).to(self.device)
            
            # 将GAIL奖励与环境奖励结合（权重可调）
            reward_batch = reward_batch + 0.5 * gail_rewards
        
        # ---- 更新Q网络 ---- #
        # 计算下一状态动作的Q值
        with torch.no_grad():
            next_actions, next_log_probs, _, _, _ = self.policy_net.evaluate(next_state_batch)
            target_q1 = self.target_q1_net(next_state_batch, next_actions)
            target_q2 = self.target_q2_net(next_state_batch, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
        
        # 计算当前Q值
        current_q1 = self.q1_net(state_batch, action_batch)
        current_q2 = self.q2_net(state_batch, action_batch)
        
        # 计算Q网络损失
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新Q网络
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # ---- 更新策略网络 ---- #
        if update_policy:
            new_actions, log_probs, _, _, _ = self.policy_net.evaluate(state_batch)
            q1 = self.q1_net(state_batch, new_actions)
            q2 = self.q2_net(state_batch, new_actions)
            q = torch.min(q1, q2)
            
            # 计算策略损失
            policy_loss = (self.alpha * log_probs - q).mean()
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 软更新目标网络
            for target_param, param in zip(self.target_q1_net.parameters(), self.q1_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for target_param, param in zip(self.target_q2_net.parameters(), self.q2_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 生成专家数据的函数
def generate_expert_data(env, num_episodes=10):
    """生成专家轨迹数据
    
    这里我们使用一个简单的启发式策略来生成专家数据
    实际应用中可以使用人类演示或更复杂的控制器
    """
    states = []
    actions = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 获取末端执行器和目标位置
            end_effector_pos = state[7:10]  # 假设状态的7-9索引是末端执行器位置
            target_pos = state[10:13]  # 假设状态的10-12索引是目标位置
            
            # 计算方向向量
            direction = np.array(target_pos) - np.array(end_effector_pos)
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance  # 归一化
            
            # 简单的启发式控制：向目标方向移动
            # 这里我们假设动作是直接控制关节角度
            action = np.zeros(7)
            
            # 简单的启发式映射：方向映射到关节动作
            # 这只是一个简化示例，实际情况下需要更复杂的逆运动学计算
            action[0] = direction[0] * 0.5  # X方向映射到第一个关节
            action[1] = direction[1] * 0.5  # Y方向映射到第二个关节
            action[2] = direction[2] * 0.5  # Z方向映射到第三个关节
            
            # 对其他关节施加小的随机动作
            action[3:] = np.random.uniform(-0.1, 0.1, 4)
            
            # 执行动作
            next_state, _, done, _ = env.step(action)
            
            # 存储状态和动作
            states.append(state)
            actions.append(action)
            
            state = next_state
    
    return np.array(states), np.array(actions)

# 训练SAC+GAIL
def train_sac_gail(env, state_dim, action_dim, num_episodes=1000, max_steps=300, batch_size=256, expert_update_freq=5, save_path=None):
    """训练SAC+GAIL算法"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 初始化SAC，调整超参数
    sac = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=512,  # 增加网络容量
        gamma=0.99,
        tau=0.02,    # 增大软更新系数，加快目标网络更新
        alpha=0.05,   # 减小熵正则化系数，减少探索增加利用
        lr=1e-3,     # 提高学习率，加快收敛
        batch_size=batch_size,
        buffer_size=2000000,  # 增加缓冲区大小
        device=device
    )
    
    # 生成专家数据
    print("Generating expert data...")
    expert_states, expert_actions = generate_expert_data(env, num_episodes=50)  # 增加专家数据量
    expert_dataset = (expert_states, expert_actions)
    
    # 初始化专家网络
    expert_network = ExpertNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=512  # 增加网络容量
    )
    
    # 记录训练信息
    rewards = []
    episode_steps = []
    success_history = []
    best_success_rate = 0.0
    evaluation_freq = 10  # 更频繁地评估
    
    print("Starting training...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_step = 0
        
        # 使用余弦退火的探索噪声
        noise_scale = max(0.4 * np.cos(np.pi * episode / num_episodes), 0.05)
        
        for step in range(max_steps):
            # 选择动作并添加噪声
            action = sac.select_action(state)
            action = np.clip(action + noise_scale * np.random.randn(action_dim), -1, 1)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储转换
            sac.store_transition(state, action, reward, next_state, done)
            
            # 更新网络
            if len(sac.replay_buffer) > batch_size:
                for _ in range(4):  # 增加每步更新次数
                    sac.update(expert_network)
            
            state = next_state
            episode_reward += reward
            episode_step += 1
            
            if done:
                break
        
        # 更新专家网络
        if episode % expert_update_freq == 0:
            for _ in range(30):  # 增加专家网络的更新次数
                policy_states = []
                policy_actions = []
                
                test_state = env.reset()
                for _ in range(300):  # 增加采样数量
                    test_action = sac.select_action(test_state)
                    next_test_state, _, test_done, _ = env.step(test_action)
                    
                    policy_states.append(test_state)
                    policy_actions.append(test_action)
                    
                    test_state = next_test_state
                    if test_done:
                        test_state = env.reset()
                
                policy_data = (np.array(policy_states), np.array(policy_actions))
                expert_network.train(expert_dataset, policy_data, iterations=30)
        
        # 记录信息
        rewards.append(episode_reward)
        episode_steps.append(episode_step)
        
        # 评估和保存模型
        if (episode + 1) % evaluation_freq == 0:
            success_rate = evaluate_policy(env, sac.policy_net, device, num_episodes=20)
            success_history.append(success_rate)
            print(f"评估 - 回合 {episode+1}: 成功率 = {success_rate:.2f}")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                torch.save({
                    'policy': sac.policy_net.state_dict(),
                    'q1': sac.q1_net.state_dict(),
                    'q2': sac.q2_net.state_dict(),
                    'target_q1': sac.target_q1_net.state_dict(),
                    'target_q2': sac.target_q2_net.state_dict(),
                    'discriminator': expert_network.state_dict()
                }, f"{save_path}/sac_gail_best.pt")
                print(f"保存了新的最佳模型，成功率: {best_success_rate:.2f}")
        
        # 打印训练信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            print(f"回合 {episode+1}/{num_episodes} | 平均奖励: {avg_reward:.2f} | 平均步数: {avg_steps:.2f}")
        
        # 定期保存检查点
        if save_path and (episode + 1) % 100 == 0:
            torch.save({
                'policy': sac.policy_net.state_dict(),
                'q1': sac.q1_net.state_dict(),
                'q2': sac.q2_net.state_dict(),
                'target_q1': sac.target_q1_net.state_dict(),
                'target_q2': sac.target_q2_net.state_dict(),
                'discriminator': expert_network.state_dict(),
                'rewards': rewards,
                'success_history': success_history
            }, f"{save_path}/sac_gail_episode_{episode+1}.pt")
    
    return sac, expert_network, rewards, success_history

# 添加一个评估函数，用于定期评估策略性能
def evaluate_policy(env, policy, device, num_episodes=10):
    """评估策略的成功率"""
    success_count = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        for _ in range(env.max_steps):
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                mean, _ = policy(state_tensor)
                action = torch.tanh(mean).cpu().numpy()
            
            next_state, _, done, info = env.step(action)
            state = next_state
            
            if done:
                if info.get('reached_target', False):
                    success_count += 1
                break
    
    return success_count / num_episodes

# 主训练脚本
if __name__ == "__main__":
    # 创建环境
    env = RobotArmEnv(render=True)
    
    # 训练参数
    state_dim = env.state_dim
    action_dim = env.action_dim
    num_episodes = 500
    max_steps = 200
    batch_size = 128
    
    # 训练SAC+GAIL
    sac, expert_network, rewards, success_history = train_sac_gail(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        num_episodes=num_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        save_path="./models"
    )
    
    # 保存最终模型
    torch.save({
        'policy': sac.policy_net.state_dict(),
        'q1': sac.q1_net.state_dict(),
        'q2': sac.q2_net.state_dict(),
        'target_q1': sac.target_q1_net.state_dict(),
        'target_q2': sac.target_q2_net.state_dict(),
        'discriminator': expert_network.state_dict(),
        'rewards': rewards,
        'success_history': success_history
    }, "./models/sac_gail_final.pt")
    
    # 关闭环境
    env.close()   