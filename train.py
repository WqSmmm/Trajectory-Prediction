import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from structure import RobotArmEnv, ReplayBuffer, PolicyNet, SoftQNet, ExpertNetwork
from collections import deque
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="用于机械臂轨迹规划的SAC+GAIL算法")
    
    # 通用参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--render", action="store_true", default=False, help="是否渲染环境")
    
    # 训练参数
    parser.add_argument("--max_steps", type=int, default=200, help="单个回合的最大步数")
    parser.add_argument("--total_episodes", type=int, default=5000, help="总训练回合数")
    parser.add_argument("--warm_up_episodes", type=int, default=10, help="收集初始数据的回合数")
    parser.add_argument("--batch_size", type=int, default=128, help="批量大小")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="折扣因子γ")
    parser.add_argument("--buffer_size", type=int, default=100000, help="回放缓冲区大小")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--soft_tau", type=float, default=0.01, help="软更新系数")
    parser.add_argument("--expert_epochs", type=int, default=10, help="每个回合专家网络训练的轮数")
    parser.add_argument("--random_steps", type=int, default=1000, help="初始随机动作的步数")
    
    # 噪声参数
    parser.add_argument("--noise_std", type=float, default=0.2, help="初始探索噪声标准差")
    parser.add_argument("--min_noise", type=float, default=0.05, help="最小探索噪声")
    parser.add_argument("--noise_decay", type=float, default=0.995, help="噪声衰减率")
    
    # 新增高级训练参数
    parser.add_argument("--alpha_lr", type=float, default=3e-4, help="熵权重学习率")
    parser.add_argument("--target_entropy", type=float, default=None, help="目标熵值，如果为None则自动设置")
    parser.add_argument("--update_interval", type=int, default=1, help="策略更新间隔(步数)")
    parser.add_argument("--expert_weight", type=float, default=0.5, help="专家奖励权重")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="奖励缩放系数")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--early_stop_patience", type=int, default=100, help="早停的耐心值(回合数)")
    parser.add_argument("--lr_patience", type=int, default=20, help="学习率调整的耐心值(回合数)")
    parser.add_argument("--reward_threshold", type=float, default=80.0, help="判定为解决的奖励阈值")
    parser.add_argument("--reset_training", type=bool, default=True, help="当停滞时重置优化器")
    
    # 保存和加载参数
    parser.add_argument("--save_interval", type=int, default=100, help="保存模型的间隔(回合数)")
    parser.add_argument("--save_dir", type=str, default="./models", help="模型保存路径")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard日志路径")
    parser.add_argument("--load_model", type=str, default=None, help="加载现有模型路径")
    parser.add_argument("--resume_training", action="store_true", default=False, help="是否从上次训练继续")
    
    # 评估参数
    parser.add_argument("--eval_interval", type=int, default=50, help="评估间隔(回合数)")
    parser.add_argument("--eval_episodes", type=int, default=5, help="每次评估的回合数")
    
    return parser.parse_args()

def collect_expert_data(env, num_trajectories=20, max_steps=200):
    """收集专家演示数据"""
    expert_states = []
    expert_actions = []
    
    for trajectory in range(num_trajectories):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # 获取末端执行器和目标位置
            end_effector_pos = state[7:10]
            target_pos = state[10:13]
            
            # 计算方向向量
            direction = target_pos - end_effector_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
            
            # 生成7维动作向量
            action = np.zeros(7)
            
            # 将方向向量映射到关节动作
            scale = 0.5  # 动作缩放因子
            action[0:3] = direction[0] * scale  # x方向影响前3个关节
            action[3:5] = direction[1] * scale  # y方向影响中间关节
            action[5:7] = direction[2] * scale  # z方向影响后2个关节
            
            # 添加一些随机扰动以增加多样性
            noise = np.random.normal(0, 0.1, 7)
            action += noise
            
            # 限制动作范围
            action = np.clip(action, -1.0, 1.0)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储状态和动作
            expert_states.append(state)
            expert_actions.append(action)
            
            state = next_state
            steps += 1
            
            # 如果到达目标，提前结束
            if info.get('reached_target', False):
                break
    
    return np.array(expert_states), np.array(expert_actions)

def preprocess_expert_data(expert_demonstrations):
    """预处理专家数据用于训练专家网络"""
    states = []
    actions = []
    
    for trajectory in expert_demonstrations:
        for transition in trajectory:
            state, action, _, _, _ = transition
            states.append(state)
            actions.append(action)
    
    return np.array(states), np.array(actions)

def train_expert_network(expert_net, states, actions, device, batch_size, num_epochs, optimizer):
    """训练专家网络拟合专家数据"""
    dataset_size = len(states)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # 随机打乱数据
        indices = np.random.permutation(dataset_size)
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            state_batch = torch.FloatTensor(states[batch_indices]).to(device)
            action_batch = torch.FloatTensor(actions[batch_indices]).to(device)
            
            # 前向传播
            predicted_actions = expert_net(state_batch)
            
            # 计算MSE损失
            loss = nn.MSELoss()(predicted_actions, action_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 2 == 0:
            print(f"Expert Network Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    return avg_loss

def eval_agent(env, policy, device, num_episodes=5, max_steps=200):
    """评估当前策略的性能"""
    total_rewards = []
    success_count = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                mean, _ = policy(state_tensor)
                action = torch.tanh(mean).squeeze(0).cpu().numpy()
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                if info.get('reached_target', False):
                    success_count += 1
                break
        
        total_rewards.append(episode_reward)
    
    mean_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes
    
    return mean_reward, success_rate

class EarlyStopping:
    """早停机制，当验证性能不再提升时停止训练"""
    def __init__(self, patience=50, delta=0.01, min_epochs=100):
        self.patience = patience
        self.delta = delta
        self.min_epochs = min_epochs  # 最小训练轮数
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.best_model = None
        self.should_stop = False
        self.training_history = []
    
    def __call__(self, score, model_dict, save_path, epoch):
        """检查是否应该早停
        
        Args:
            score: 当前评分
            model_dict: 模型状态字典
            save_path: 保存路径
            epoch: 当前轮数
        """
        self.training_history.append(score)
        
        if epoch < self.min_epochs:  # 确保最小训练轮数
            return False
        
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model_dict, save_path)
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n早停触发! 在第 {self.best_epoch} 轮达到最佳分数: {self.best_score:.4f}")
                print(f"之后 {self.patience} 轮未见改善")
                return True
            return False
    
    def save_checkpoint(self, model_dict, save_path):
        """保存最佳模型"""
        self.best_model = model_dict
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state': model_dict,
            'best_score': self.best_score,
            'epoch': self.best_epoch,
            'training_history': self.training_history
        }, save_path)
        print(f"\n保存最佳模型 - 分数: {self.best_score:.4f}")

def reset_optimizer(optimizer, model):
    """重置优化器状态"""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    return optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])

def collect_dagger_data(env, expert_policy, policy_net, device, beta, num_trajectories=20, max_steps=200):
    """使用DAGGER算法收集数据
    
    Args:
        env: 环境
        expert_policy: 专家策略
        policy_net: 当前策略
        device: 计算设备
        beta: 专家策略使用概率
        num_trajectories: 轨迹数量
        max_steps: 最大步数
    """
    states = []
    expert_actions = []
    
    for trajectory in range(num_trajectories):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # 专家演示和学习者混合
            if np.random.random() < beta:
                # 使用专家策略
                end_effector_pos = state[7:10]
                target_pos = state[10:13]
                direction = target_pos - end_effector_pos
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    direction = direction / distance
                
                action = np.zeros(7)
                scale = 0.5
                action[0:3] = direction[0] * scale
                action[3:5] = direction[1] * scale
                action[5:7] = direction[2] * scale
                noise = np.random.normal(0, 0.1, 7)
                action = np.clip(action + noise, -1.0, 1.0)
            else:
                # 使用当前策略
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    mean, _ = policy_net(state_tensor)
                    action = torch.tanh(mean).squeeze(0).cpu().numpy()
            
            # 获取专家动作用于训练
            end_effector_pos = state[7:10]
            target_pos = state[10:13]
            direction = target_pos - end_effector_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
            
            expert_action = np.zeros(7)
            scale = 0.5
            expert_action[0:3] = direction[0] * scale
            expert_action[3:5] = direction[1] * scale
            expert_action[5:7] = direction[2] * scale
            expert_noise = np.random.normal(0, 0.05, 7)  # 减小专家噪声
            expert_action = np.clip(expert_action + expert_noise, -1.0, 1.0)
            
            # 存储状态和专家动作
            states.append(state)
            expert_actions.append(expert_action)
            
            # 执行实际动作
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1
            
            if info.get('reached_target', False):
                break
    
    return np.array(states), np.array(expert_actions)

def main():
    """主训练循环"""
    args = parse_args()
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化环境
    env = RobotArmEnv(render=args.render, max_steps=args.max_steps)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # 初始化策略网络和Q网络
    policy_net = PolicyNet(state_dim, action_dim, args.hidden_dim).to(device)
    q_net1 = SoftQNet(state_dim, action_dim, args.hidden_dim).to(device)
    q_net2 = SoftQNet(state_dim, action_dim, args.hidden_dim).to(device)
    target_q_net1 = SoftQNet(state_dim, action_dim, args.hidden_dim).to(device)
    target_q_net2 = SoftQNet(state_dim, action_dim, args.hidden_dim).to(device)
    
    # 复制参数到目标网络
    for target_param, param in zip(target_q_net1.parameters(), q_net1.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(target_q_net2.parameters(), q_net2.parameters()):
        target_param.data.copy_(param.data)
    
    # 初始化专家网络
    expert_net = ExpertNetwork(state_dim, action_dim, args.hidden_dim).to(device)
    
    # 初始化优化器
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    q1_optimizer = optim.Adam(q_net1.parameters(), lr=args.learning_rate)
    q2_optimizer = optim.Adam(q_net2.parameters(), lr=args.learning_rate)
    expert_optimizer = optim.Adam(expert_net.parameters(), lr=args.learning_rate)
    
    # 初始化学习率调度器
    policy_scheduler = ReduceLROnPlateau(policy_optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    q1_scheduler = ReduceLROnPlateau(q1_optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    q2_scheduler = ReduceLROnPlateau(q2_optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    
    # 初始化回放缓冲区
    replay_buffer = ReplayBuffer(args.buffer_size)
    
    # 初始化TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # 收集和预处理专家数据
    expert_states, expert_actions = collect_expert_data(env, num_trajectories=20, max_steps=args.max_steps)
    
    # 训练专家网络
    print("训练专家网络...")
    train_expert_network(expert_net, expert_states, expert_actions, device, args.batch_size, args.expert_epochs, expert_optimizer)
    
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=args.early_stop_patience, delta=0.01, min_epochs=100)
    
    # 初始化训练统计
    episode_rewards = deque(maxlen=100)
    best_reward = float('-inf')
    noise_std = args.noise_std
    
    # 主训练循环
    print("开始训练...")
    for episode in range(args.total_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < args.max_steps:
            # 将状态转换为张量
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # 根据当前策略选择动作
            with torch.no_grad():
                mean, log_std = policy_net(state_tensor)
                std = log_std.exp()
                normal = Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
                action = action.squeeze(0).cpu().numpy()
            
            # 添加探索噪声
            noise = np.random.normal(0, noise_std, size=action_dim)
            action = np.clip(action + noise, -1.0, 1.0)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储转换
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # 当回放缓冲区中有足够的样本时更新网络
            if len(replay_buffer) > args.batch_size:
                # 从回放缓冲区采样
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(args.batch_size)
                
                # 转换为张量
                state_batch = torch.FloatTensor(state_batch).to(device)
                action_batch = torch.FloatTensor(action_batch).to(device)
                reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(device)
                done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
                
                # 更新Q网络
                with torch.no_grad():
                    next_state_action, next_state_log_pi = policy_net.sample(next_state_batch)
                    target_q1 = target_q_net1(next_state_batch, next_state_action)
                    target_q2 = target_q_net2(next_state_batch, next_state_action)
                    target_q = torch.min(target_q1, target_q2) - next_state_log_pi
                    target_q = reward_batch + (1 - done_batch) * args.discount_factor * target_q
                
                # Q1损失
                current_q1 = q_net1(state_batch, action_batch)
                q1_loss = F.mse_loss(current_q1, target_q)
                
                # Q2损失
                current_q2 = q_net2(state_batch, action_batch)
                q2_loss = F.mse_loss(current_q2, target_q)
                
                # 更新Q网络
                q1_optimizer.zero_grad()
                q1_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net1.parameters(), args.clip_grad)
                q1_optimizer.step()
                
                q2_optimizer.zero_grad()
                q2_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net2.parameters(), args.clip_grad)
                q2_optimizer.step()
                
                # 延迟策略更新
                if steps % args.update_interval == 0:
                    # 计算策略损失
                    new_actions, log_pi = policy_net.sample(state_batch)
                    q1_new = q_net1(state_batch, new_actions)
                    q2_new = q_net2(state_batch, new_actions)
                    q_new = torch.min(q1_new, q2_new)
                    
                    policy_loss = (log_pi - q_new).mean()
                    
                    # 更新策略网络
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.clip_grad)
                    policy_optimizer.step()
                    
                    # 软更新目标网络
                    for target_param, param in zip(target_q_net1.parameters(), q_net1.parameters()):
                        target_param.data.copy_(args.soft_tau * param.data + (1 - args.soft_tau) * target_param.data)
                    for target_param, param in zip(target_q_net2.parameters(), q_net2.parameters()):
                        target_param.data.copy_(args.soft_tau * param.data + (1 - args.soft_tau) * target_param.data)
        
        # 更新噪声
        noise_std = max(args.min_noise, noise_std * args.noise_decay)
        
        # 记录奖励
        episode_rewards.append(episode_reward)
        mean_reward = np.mean(episode_rewards)
        
        # 评估当前策略
        if (episode + 1) % args.eval_interval == 0:
            eval_reward, success_rate = eval_agent(env, policy_net, device, args.eval_episodes, args.max_steps)
            print(f"Episode {episode+1}: Eval Reward: {eval_reward:.2f}, Success Rate: {success_rate:.2%}")
            
            # 更新学习率
            policy_scheduler.step(eval_reward)
            q1_scheduler.step(eval_reward)
            q2_scheduler.step(eval_reward)
            
            # 记录到TensorBoard
            writer.add_scalar('Eval/Average_Reward', eval_reward, episode)
            writer.add_scalar('Eval/Success_Rate', success_rate, episode)
            
            # 检查是否需要早停
            if early_stopping(eval_reward, {
                'policy_state_dict': policy_net.state_dict(),
                'q1_state_dict': q_net1.state_dict(),
                'q2_state_dict': q_net2.state_dict(),
                'target_q1_state_dict': target_q_net1.state_dict(),
                'target_q2_state_dict': target_q_net2.state_dict(),
                'expert_state_dict': expert_net.state_dict()
            }, os.path.join(args.save_dir, 'best_model.pth'), episode):
                print("Early stopping triggered!")
                break
        
        # 保存模型
        if (episode + 1) % args.save_interval == 0:
            torch.save({
                'episode': episode,
                'policy_state_dict': policy_net.state_dict(),
                'q1_state_dict': q_net1.state_dict(),
                'q2_state_dict': q_net2.state_dict(),
                'target_q1_state_dict': target_q_net1.state_dict(),
                'target_q2_state_dict': target_q_net2.state_dict(),
                'expert_state_dict': expert_net.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'q1_optimizer_state_dict': q1_optimizer.state_dict(),
                'q2_optimizer_state_dict': q2_optimizer.state_dict(),
                'expert_optimizer_state_dict': expert_optimizer.state_dict()
            }, os.path.join(args.save_dir, f'checkpoint_{episode+1}.pth'))
        
        # 打印训练信息
        print(f"Episode {episode+1}: Reward: {episode_reward:.2f}, Average Reward: {mean_reward:.2f}")
        writer.add_scalar('Train/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Train/Average_Reward', mean_reward, episode)
        writer.add_scalar('Train/Noise_STD', noise_std, episode)
    
    # 关闭环境和TensorBoard
    env.close()
    writer.close()

if __name__ == "__main__":
    main() 