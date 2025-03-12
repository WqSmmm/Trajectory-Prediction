import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import seaborn as sns
from structure import RobotArmEnv, PolicyNet, SoftQNet

def parse_args():
    parser = argparse.ArgumentParser(description="评估机械臂轨迹规划模型")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")
    parser.add_argument("--max_steps", type=int, default=200, help="每个回合的最大步数")
    parser.add_argument("--num_episodes", type=int, default=10, help="评估回合数")
    parser.add_argument("--save_path", type=str, default="./evaluation_results", help="评估结果保存路径")
    parser.add_argument("--record_video", action="store_true", help="是否记录视频")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--compare_models", action="store_true", help="是否比较多个模型")
    return parser.parse_args()

def visualize_trajectory(env, states, rewards, success, path, episode_num, extra_info=None):
    """可视化单条轨迹"""
    # 提取末端执行器位置
    end_effector_positions = [state[7:10] for state in states]  # 假设状态的7-9索引是末端执行器位置
    end_effector_positions = np.array(end_effector_positions)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(231, projection='3d')
    
    # 绘制目标点
    target_pos = states[0][10:13]  # 假设状态的10-12索引是目标位置
    ax1.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], 
               color='red', s=100, marker='*', label='目标点')
    
    # 绘制障碍物
    for obstacle in env.obstacles:
        pos = obstacle['position']
        ax1.scatter([pos[0]], [pos[1]], [pos[2]], 
                   color='black', s=100, marker='o', label='障碍物')
    
    # 绘制轨迹，根据成功与否改变颜色
    color = 'green' if success else 'red'
    ax1.plot(end_effector_positions[:, 0], end_effector_positions[:, 1], end_effector_positions[:, 2], 
             color=color, linewidth=2)
    
    # 绘制起点和终点
    ax1.scatter([end_effector_positions[0, 0]], [end_effector_positions[0, 1]], [end_effector_positions[0, 2]],
                color='blue', s=50, label='起点')
    ax1.scatter([end_effector_positions[-1, 0]], [end_effector_positions[-1, 1]], [end_effector_positions[-1, 2]],
                color=color, s=50, label='终点')
    
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_zlabel('Z', fontsize=12)
    ax1.set_title(f'轨迹 ({"成功" if success else "失败"})', fontsize=14)
    ax1.legend(fontsize=10)
    
    # 同一轨迹的另一个视角
    ax2 = fig.add_subplot(232, projection='3d')
    
    # 复制相同的内容
    ax2.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], 
               color='red', s=100, marker='*', label='目标点')
    
    for obstacle in env.obstacles:
        pos = obstacle['position']
        ax2.scatter([pos[0]], [pos[1]], [pos[2]], 
                   color='black', s=100, marker='o', label='障碍物')
    
    ax2.plot(end_effector_positions[:, 0], end_effector_positions[:, 1], end_effector_positions[:, 2], 
             color=color, linewidth=2)
    
    ax2.scatter([end_effector_positions[0, 0]], [end_effector_positions[0, 1]], [end_effector_positions[0, 2]],
                color='blue', s=50, label='起点')
    ax2.scatter([end_effector_positions[-1, 0]], [end_effector_positions[-1, 1]], [end_effector_positions[-1, 2]],
                color=color, s=50, label='终点')
    
    # 设置不同的视角
    ax2.view_init(elev=20, azim=70)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Z', fontsize=12)
    ax2.set_title('轨迹（另一视角）', fontsize=14)
    
    # 奖励曲线
    ax3 = fig.add_subplot(233)
    ax3.plot(rewards, marker='o', linestyle='-', markersize=3, color='blue')
    ax3.set_xlabel('步数', fontsize=12)
    ax3.set_ylabel('奖励', fontsize=12)
    ax3.set_title('每步奖励', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 添加累积奖励信息
    total_reward = sum(rewards)
    ax3.text(0.5, 0.95, f'总奖励: {total_reward:.2f}', 
             transform=ax3.transAxes, ha='center', va='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=12)
    
    # 距离到目标的变化
    ax4 = fig.add_subplot(234)
    distances = [np.linalg.norm(np.array(end_effector_positions[i]) - np.array(target_pos)) 
                for i in range(len(end_effector_positions))]
    
    ax4.plot(distances, marker='o', linestyle='-', markersize=3, color='purple')
    ax4.set_xlabel('步数', fontsize=12)
    ax4.set_ylabel('到目标距离', fontsize=12)
    ax4.set_title('到目标的距离变化', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # 速度变化图
    ax5 = fig.add_subplot(235)
    velocities = []
    for i in range(1, len(end_effector_positions)):
        velocity = np.linalg.norm(end_effector_positions[i] - end_effector_positions[i-1])
        velocities.append(velocity)
    
    if velocities:
        ax5.plot(velocities, marker='o', linestyle='-', markersize=3, color='orange')
        ax5.set_xlabel('步数', fontsize=12)
        ax5.set_ylabel('速度', fontsize=12)
        ax5.set_title('末端执行器速度变化', fontsize=14)
        ax5.grid(True, alpha=0.3)
    
    # 附加信息图表
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    info_text = f"回合: {episode_num}\n"
    info_text += f"步数: {len(states)-1}\n"
    info_text += f"结果: {'成功' if success else '失败'}\n"
    info_text += f"总奖励: {total_reward:.2f}\n"
    info_text += f"起点到目标距离: {distances[0]:.4f}\n"
    info_text += f"终点到目标距离: {distances[-1]:.4f}\n"
    
    if extra_info:
        for key, value in extra_info.items():
            info_text += f"{key}: {value}\n"
    
    ax6.text(0.1, 0.5, info_text, fontsize=12, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'trajectory_episode_{episode_num}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(env, policy, device, num_episodes=10, max_steps=200):
    """评估模型性能"""
    total_rewards = []
    success_count = 0
    episode_steps = []
    final_distances = []
    smoothness_scores = []
    collision_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        positions = []  # 记录轨迹点
        
        for step in range(max_steps):
            # 将状态转换为张量
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # 获取动作
            with torch.no_grad():
                mean, _ = policy(state_tensor)
                action = torch.tanh(mean).squeeze(0).cpu().numpy()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录轨迹点
            positions.append(next_state[7:10])  # 假设7:10是末端执行器位置
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                if info.get('reached_target', False):
                    success_count += 1
                if info.get('collision', False):
                    collision_count += 1
                break
        
        # 计算最终距离
        final_pos = positions[-1]
        target_pos = state[10:13]  # 假设10:13是目标位置
        final_distance = np.linalg.norm(final_pos - target_pos)
        
        # 计算轨迹平滑度（使用加速度的平方和）
        positions = np.array(positions)
        if len(positions) > 2:
            velocities = np.diff(positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            smoothness = -np.sum(accelerations**2)  # 负的加速度平方和
            smoothness_scores.append(smoothness)
        
        total_rewards.append(episode_reward)
        episode_steps.append(step_count)
        final_distances.append(final_distance)
        
        print(f"回合 {episode+1}/{num_episodes}:")
        print(f"  奖励: {episode_reward:.2f}")
        print(f"  步数: {step_count}")
        print(f"  最终距离: {final_distance:.4f}")
        print(f"  是否成功: {'是' if info.get('reached_target', False) else '否'}")
        print(f"  是否碰撞: {'是' if info.get('collision', False) else '否'}")
        print("------------------------")
    
    # 计算统计数据
    success_rate = success_count / num_episodes
    collision_rate = collision_count / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(episode_steps)
    avg_final_distance = np.mean(final_distances)
    avg_smoothness = np.mean(smoothness_scores)
    
    results = {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_final_distance': avg_final_distance,
        'avg_smoothness': avg_smoothness,
        'total_rewards': total_rewards,
        'episode_steps': episode_steps,
        'final_distances': final_distances,
        'smoothness_scores': smoothness_scores
    }
    
    return results

def print_evaluation_results(results):
    """打印评估结果"""
    print("\n========= 评估结果 =========")
    print(f"成功率: {results['success_rate']:.2%}")
    print(f"碰撞率: {results['collision_rate']:.2%}")
    print(f"平均奖励: {results['avg_reward']:.2f}")
    print(f"平均步数: {results['avg_steps']:.2f}")
    print(f"平均最终距离: {results['avg_final_distance']:.4f}")
    print(f"平均轨迹平滑度: {results['avg_smoothness']:.4f}")
    print("===========================")

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = RobotArmEnv(render=args.render, max_steps=args.max_steps)
    
    # 获取状态和动作维度
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    try:
        # 加载模型
        checkpoint = torch.load("./models/sac_gail_episode_50.pt", map_location=device)
        
        # 创建策略网络并加载权重
        policy_net = PolicyNet(state_dim, action_dim, args.hidden_dim).to(device)
        policy_net.load_state_dict(checkpoint['policy'])
        policy_net.eval()
        
        print("模型加载成功，开始评估...")
        
        # 评估模型
        results = evaluate_model(
            env=env,
            policy=policy_net,
            device=device,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps
        )
        
        # 打印结果
        print_evaluation_results(results)
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
    finally:
        env.close()

if __name__ == "__main__":
    main() 