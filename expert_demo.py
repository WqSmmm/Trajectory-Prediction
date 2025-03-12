import numpy as np
import torch
import os
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from structure import RobotArmEnv, PolicyNet
import seaborn as sns

def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_expert_trajectory(env, num_episodes=30, max_steps=200, render=True, save_path="./expert_data"):
    """
    使用预定义的控制规则生成专家轨迹
    
    参数:
        env: 机械臂环境
        num_episodes: 收集的轨迹数量
        max_steps: 每个轨迹的最大步数
        render: 是否渲染
        save_path: 保存轨迹数据的路径
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    all_trajectories = []  # 用于可视化
    successful_episodes = 0  # 记录成功到达目标的回合数
    
    for episode in range(num_episodes):
        print(f"采集轨迹 {episode+1}/{num_episodes}")
        state = env.reset()
        trajectory = []  # 记录当前轨迹中的末端执行器位置
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []
        success = False
        
        # 记录初始位置
        end_effector_pos = state[7:10]  # 假设状态的7-9索引是末端执行器位置
        trajectory.append(end_effector_pos)
        
        for step in range(max_steps):
            # 获取末端执行器和目标位置
            end_effector_pos = state[7:10]  # 末端执行器位置
            target_pos = state[10:13]  # 目标位置
            
            # 计算方向向量（直接朝向目标）
            direction = np.array(target_pos) - np.array(end_effector_pos)
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance  # 归一化
            
            # 智能避障：检测是否需要避开障碍物
            obstacle_features = state[13:]
            obstacle_influence = np.zeros(3)
            
            # 解析障碍物特征（每个障碍物有3个方向分量）
            for i in range(0, min(len(obstacle_features), 15), 3):
                if np.any(obstacle_features[i:i+3] != 0):  # 如果有障碍物特征
                    # 障碍物方向向量（已在特征中归一化）
                    obstacle_dir = obstacle_features[i:i+3]
                    
                    # 根据障碍物方向添加斥力
                    # 如果障碍物方向与目标方向相反，减弱避障
                    dot_product = np.dot(obstacle_dir, direction)
                    
                    # 如果障碍物在运动路径上（方向相似），则强化避障
                    if dot_product > 0.7:  # 方向相似度高
                        repulsion = obstacle_dir * 0.2  # 强化避障力
                    else:
                        repulsion = obstacle_dir * 0.1  # 正常避障力
                    
                    obstacle_influence += repulsion
            
            # 整合目标导向和避障导向
            combined_direction = direction - obstacle_influence
            
            # 归一化组合方向
            if np.linalg.norm(combined_direction) > 0:
                combined_direction = combined_direction / np.linalg.norm(combined_direction)
            
            # 自适应速度控制
            # 距离目标近时减小动作幅度，远时增大
            speed_scale = min(0.8, max(0.2, distance / 1.5))
            
            # 平滑动作：创建3维动作 (dx, dy, dz)
            action = combined_direction * speed_scale
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储当前步的数据
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_states.append(next_state)
            episode_dones.append(done)
            
            # 记录末端执行器位置用于可视化
            next_end_effector_pos = next_state[7:10]
            trajectory.append(next_end_effector_pos)
            
            # 更新状态
            state = next_state
            
            if done:
                # 如果成功完成任务
                if info.get('reached_target', False):
                    success = True
                    successful_episodes += 1
                break
        
        # 只保存成功的轨迹
        if success:
            all_states.extend(episode_states)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            all_next_states.extend(episode_next_states)
            all_dones.extend(episode_dones)
            all_trajectories.append(trajectory)
            print(f"  成功轨迹，步数: {len(trajectory)-1}")
        else:
            print(f"  失败轨迹，丢弃")
            
    success_rate = successful_episodes / num_episodes
    print(f"成功率: {success_rate:.2f} ({successful_episodes}/{num_episodes})")
    
    # 数据增强：增加数据多样性
    if len(all_states) > 0:
        print("执行数据增强...")
        augmented_states = []
        augmented_actions = []
        augmented_rewards = []
        augmented_next_states = []
        augmented_dones = []
        
        for i in range(len(all_states)):
            # 原始数据
            augmented_states.append(all_states[i])
            augmented_actions.append(all_actions[i])
            augmented_rewards.append(all_rewards[i])
            augmented_next_states.append(all_next_states[i])
            augmented_dones.append(all_dones[i])
            
            # 添加带噪声的样本，但只对非终止状态进行
            if not all_dones[i]:
                for _ in range(2):  # 每个样本增强2个变体
                    # 状态噪声
                    noise_state = all_states[i].copy()
                    # 只对关节角度和末端执行器位置添加噪声
                    noise_state[:10] += np.random.normal(0, 0.01, size=10)
                    
                    # 动作噪声
                    noise_action = all_actions[i] + np.random.normal(0, 0.05, size=len(all_actions[i]))
                    noise_action = np.clip(noise_action, -1, 1)  # 确保动作在范围内
                    
                    # 奖励可能会略有不同，但保持差不多
                    noise_reward = all_rewards[i] + np.random.normal(0, 0.1)
                    
                    # 下一个状态也需要对应变化
                    noise_next_state = all_next_states[i].copy()
                    noise_next_state[:10] += np.random.normal(0, 0.01, size=10)
                    
                    # 完成状态不变
                    noise_done = all_dones[i]
                    
                    augmented_states.append(noise_state)
                    augmented_actions.append(noise_action)
                    augmented_rewards.append(noise_reward)
                    augmented_next_states.append(noise_next_state)
                    augmented_dones.append(noise_done)
        
        all_states = augmented_states
        all_actions = augmented_actions
        all_rewards = augmented_rewards
        all_next_states = augmented_next_states
        all_dones = augmented_dones
        
        print(f"数据增强后的样本数: {len(all_states)}")
    
    # 保存专家数据
    np.save(os.path.join(save_path, "expert_states.npy"), np.array(all_states))
    np.save(os.path.join(save_path, "expert_actions.npy"), np.array(all_actions))
    np.save(os.path.join(save_path, "expert_rewards.npy"), np.array(all_rewards))
    np.save(os.path.join(save_path, "expert_next_states.npy"), np.array(all_next_states))
    np.save(os.path.join(save_path, "expert_dones.npy"), np.array(all_dones))
    
    # 可视化轨迹
    if all_trajectories:
        visualize_trajectories(all_trajectories, env, save_path)
    
    print(f"总共收集了 {len(all_states)} 个状态-动作对")
    print(f"成功轨迹数量: {len(all_trajectories)}")
    
    return all_states, all_actions

def visualize_trajectories(trajectories, env, save_path):
    """可视化轨迹"""
    # 获取所有障碍物位置和目标位置
    target_position = env.target_pos
    obstacles = env.obstacles
    
    # 绘制3D视图
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制目标点
    ax.scatter([target_position[0]], [target_position[1]], [target_position[2]], 
               color='red', s=150, marker='*', label='目标')
    
    # 绘制障碍物
    for obstacle in obstacles:
        pos = obstacle['position']
        radius = obstacle['radius']
        ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                  color='black', s=100*radius/0.1, marker='o', label='障碍物')
    
    # 计算起点和终点位置，用于绘制参考线
    all_starts = []
    all_ends = []
    
    # 创建一个颜色映射
    cmap = plt.cm.viridis
    
    # 绘制所有轨迹
    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        
        if len(traj) > 0:
            all_starts.append(traj[0])
            all_ends.append(traj[-1])
            
            # 规范化轨迹序号以获取不同颜色
            color = cmap(i / len(trajectories))
            
            # 绘制轨迹线
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color=color, 
                   linewidth=2,
                   alpha=0.7,
                   label=f'轨迹 {i+1}')
            
            # 绘制起点和终点
            ax.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], color=color, marker='^', s=80)
            ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], color=color, marker='o', s=80)
    
    # 绘制工作空间边界
    bounds = env.workspace_bounds
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    
    # 绘制工作空间边界框
    r = [x_min, x_max]
    s = [y_min, y_max]
    t = [z_min, z_max]
    
    ax.plot([r[0], r[1]], [s[0], s[0]], [t[0], t[0]], 'k-', alpha=0.2)
    ax.plot([r[0], r[1]], [s[1], s[1]], [t[0], t[0]], 'k-', alpha=0.2)
    ax.plot([r[0], r[0]], [s[0], s[1]], [t[0], t[0]], 'k-', alpha=0.2)
    ax.plot([r[1], r[1]], [s[0], s[1]], [t[0], t[0]], 'k-', alpha=0.2)
    
    ax.plot([r[0], r[1]], [s[0], s[0]], [t[1], t[1]], 'k-', alpha=0.2)
    ax.plot([r[0], r[1]], [s[1], s[1]], [t[1], t[1]], 'k-', alpha=0.2)
    ax.plot([r[0], r[0]], [s[0], s[1]], [t[1], t[1]], 'k-', alpha=0.2)
    ax.plot([r[1], r[1]], [s[0], s[1]], [t[1], t[1]], 'k-', alpha=0.2)
    
    ax.plot([r[0], r[0]], [s[0], s[0]], [t[0], t[1]], 'k-', alpha=0.2)
    ax.plot([r[1], r[1]], [s[0], s[0]], [t[0], t[1]], 'k-', alpha=0.2)
    ax.plot([r[0], r[0]], [s[1], s[1]], [t[0], t[1]], 'k-', alpha=0.2)
    ax.plot([r[1], r[1]], [s[1], s[1]], [t[0], t[1]], 'k-', alpha=0.2)
    
    ax.set_xlabel('X轴', fontsize=14)
    ax.set_ylabel('Y轴', fontsize=14)
    ax.set_zlabel('Z轴', fontsize=14)
    ax.set_title('专家轨迹可视化', fontsize=16)
    
    # 设置轴的范围
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # 添加图例，去除重复项
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    
    plt.savefig(os.path.join(save_path, "expert_trajectories_3d.png"), dpi=300, bbox_inches='tight')
    
    # 生成不同角度的视图
    for angle in [0, 45, 90, 135]:
        ax.view_init(elev=20, azim=angle)
        plt.savefig(os.path.join(save_path, f"expert_trajectories_angle_{angle}.png"), dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # 分析轨迹特点
    analyze_trajectories(trajectories, target_position, save_path)

def analyze_trajectories(trajectories, target_position, save_path):
    """分析轨迹特点并可视化"""
    if not trajectories:
        return
    
    # 轨迹长度分布
    traj_lengths = [len(traj)-1 for traj in trajectories]
    
    # 距离变化
    distance_profiles = []
    velocity_profiles = []
    smoothness_metrics = []
    
    for traj in trajectories:
        traj = np.array(traj)
        
        # 到目标的距离变化
        distances = [np.linalg.norm(pos - target_position) for pos in traj]
        distance_profiles.append(distances)
        
        # 速度分析
        velocities = []
        for i in range(1, len(traj)):
            vel = np.linalg.norm(traj[i] - traj[i-1])
            velocities.append(vel)
        velocity_profiles.append(velocities)
        
        # 平滑度分析（加速度变化）
        if len(traj) > 2:
            accelerations = []
            for i in range(1, len(velocities)):
                acc = abs(velocities[i] - velocities[i-1])
                accelerations.append(acc)
            
            # 平滑度指标：加速度的均值（越小越平滑）
            smoothness = np.mean(accelerations) if accelerations else 0
            smoothness_metrics.append(smoothness)
    
    # 创建多图布局
    fig = plt.figure(figsize=(20, 15))
    
    # 轨迹长度分布
    ax1 = fig.add_subplot(231)
    sns.histplot(traj_lengths, bins=10, kde=True, ax=ax1)
    ax1.set_xlabel('轨迹长度（步）', fontsize=12)
    ax1.set_ylabel('频次', fontsize=12)
    ax1.set_title('轨迹长度分布', fontsize=14)
    ax1.axvline(x=np.mean(traj_lengths), color='r', linestyle='--', 
               label=f'平均: {np.mean(traj_lengths):.1f}步')
    ax1.legend()
    
    # 距离变化
    ax2 = fig.add_subplot(232)
    for i, dist in enumerate(distance_profiles):
        steps = range(len(dist))
        label = f'轨迹 {i+1}' if i < 5 else None  # 只标出前5条轨迹
        alpha = 0.8 if i < 5 else 0.3
        ax2.plot(steps, dist, alpha=alpha, label=label)
    
    ax2.set_xlabel('步数', fontsize=12)
    ax2.set_ylabel('到目标距离', fontsize=12)
    ax2.set_title('到目标距离变化', fontsize=14)
    if len(distance_profiles) > 5:
        ax2.legend(loc='upper right')
    else:
        ax2.legend()
    
    # 速度分析
    ax3 = fig.add_subplot(233)
    for i, vel in enumerate(velocity_profiles):
        if len(vel) > 0:
            steps = range(len(vel))
            label = f'轨迹 {i+1}' if i < 5 else None
            alpha = 0.8 if i < 5 else 0.3
            ax3.plot(steps, vel, alpha=alpha, label=label)
    
    ax3.set_xlabel('步数', fontsize=12)
    ax3.set_ylabel('速度', fontsize=12)
    ax3.set_title('轨迹速度分析', fontsize=14)
    if len(velocity_profiles) > 5:
        ax3.legend(loc='upper right')
    else:
        ax3.legend()
    
    # 平均距离变化曲线
    ax4 = fig.add_subplot(234)
    # 找出最长的轨迹长度
    max_len = max(len(dist) for dist in distance_profiles)
    
    # 初始化平均距离数组
    avg_distances = np.zeros(max_len)
    counts = np.zeros(max_len)
    
    # 累加所有距离
    for dist in distance_profiles:
        for i, d in enumerate(dist):
            avg_distances[i] += d
            counts[i] += 1
    
    # 计算平均值
    for i in range(max_len):
        if counts[i] > 0:
            avg_distances[i] /= counts[i]
    
    # 绘制平均距离曲线
    steps = range(max_len)
    ax4.plot(steps, avg_distances, 'b-', linewidth=2)
    ax4.fill_between(steps, avg_distances * 0.8, avg_distances * 1.2, alpha=0.2, color='b')
    
    ax4.set_xlabel('步数', fontsize=12)
    ax4.set_ylabel('平均距离', fontsize=12)
    ax4.set_title('平均距离变化曲线', fontsize=14)
    
    # 平滑度分布
    ax5 = fig.add_subplot(235)
    if smoothness_metrics:
        sns.histplot(smoothness_metrics, bins=8, kde=True, ax=ax5)
        ax5.set_xlabel('平滑度指标', fontsize=12)
        ax5.set_ylabel('频次', fontsize=12)
        ax5.set_title('轨迹平滑度分布', fontsize=14)
        ax5.axvline(x=np.mean(smoothness_metrics), color='r', linestyle='--', 
                   label=f'平均: {np.mean(smoothness_metrics):.4f}')
        ax5.legend()
    
    # 轨迹统计信息
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    info_text = "专家轨迹统计信息\n"
    info_text += "=" * 40 + "\n"
    info_text += f"轨迹总数: {len(trajectories)}\n"
    info_text += f"平均轨迹长度: {np.mean(traj_lengths):.2f} 步\n"
    info_text += f"最短轨迹: {min(traj_lengths)} 步\n"
    info_text += f"最长轨迹: {max(traj_lengths)} 步\n"
    info_text += f"平均初始距离: {np.mean([dist[0] for dist in distance_profiles]):.4f}\n"
    info_text += f"平均最终距离: {np.mean([dist[-1] for dist in distance_profiles]):.4f}\n"
    
    if velocity_profiles:
        avg_velocity = np.mean([np.mean(vel) for vel in velocity_profiles if len(vel) > 0])
        max_velocity = np.max([np.max(vel) for vel in velocity_profiles if len(vel) > 0])
        info_text += f"平均速度: {avg_velocity:.4f}\n"
        info_text += f"最大速度: {max_velocity:.4f}\n"
    
    if smoothness_metrics:
        info_text += f"平均平滑度: {np.mean(smoothness_metrics):.4f}\n"
        info_text += f"最佳平滑度: {min(smoothness_metrics):.4f}\n"
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
             fontsize=12, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "expert_trajectories_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    set_seed(42)  # 设置随机种子
    
    # 创建环境
    env = RobotArmEnv(render=False, max_steps=200)
    
    # 收集专家轨迹
    create_expert_trajectory(
        env=env, 
        num_episodes=50,  # 增加轨迹数量以提高数据质量
        max_steps=200, 
        render=False, 
        save_path="./expert_data"
    )
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main() 