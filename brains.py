import environment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def get_obs(env: environment.WalkerEnv) -> np.ndarray:
    """Get normalized observation vector from environment."""
    vertices1, vertices2, joint_pos = env.robot.get_state()
    door_x, door_w, door_h = env.door
    dx = (door_x - door_w * 0.5) - joint_pos[0]
    dy = joint_pos[1] - door_h if joint_pos[1] > door_h else 0
    points = env.points
    obs = np.array([
        dx, dy,
        *[coord for point in points for coord in point],
        *[coord for v in vertices1[1:3] for coord in v],
        *[coord for v in vertices2[1:3] for coord in v],
    ], dtype=np.float32)
    obs = obs / np.array([max(env.W, env.H)] * len(obs), dtype=np.float32)
    vel_norm = env.W / 10.0
    velocities = np.array([env.robot.body1.velocity, env.robot.body2.velocity], dtype=np.float32) / vel_norm
    angular_velocities = np.array([env.robot.body1.angular_velocity, env.robot.body2.angular_velocity], dtype=np.float32) / 2.0
    obs = np.concatenate([obs, velocities.flatten(), angular_velocities.flatten()])
    return obs

def get_distance_to_door(env: environment.WalkerEnv) -> float:
    """Compute distance from robot joint to door opening."""
    joint_pos = env.robot.get_joint_world_pos()
    door_x, door_w, door_h = env.door
    dx = np.abs(joint_pos[0] - door_x) - door_w * 0.5
    dx = max(dx, 0)  # Only positive if outside door width
    dy = joint_pos[1] - door_h if joint_pos[1] > door_h else 0
    return np.sqrt(dx**2 + dy**2)

def calculate_reward(
    env: environment.WalkerEnv,
    prev_dist: float,
    time_scale: float = 0.0005, 
    progress_scale: float = 0.015,
    angular_velocities_scale: float = 0.0001) -> tuple[float, float, dict]:
    """Calculate reward using progress-based distance reduction.
    
    Returns:
        reward: The computed reward
        current_dist: Current distance to door (for next step's prev_dist)
        reward_components: Dictionary with reward components for logging
    """
    current_dist = get_distance_to_door(env)
    r_progress = (prev_dist - current_dist) * progress_scale
    r_time = -time_scale * env.time_count
    angular_velocities = np.abs(env.robot.body1.angular_velocity) + np.abs(env.robot.body2.angular_velocity)
    r_angular = -angular_velocities_scale * angular_velocities
    r_goal = 100.0 if reached_goal(env) else 0.0
    reward = r_progress + r_time + r_angular + r_goal
    
    reward_components = {
        'progress': r_progress,
        'time': r_time,
        'angular': r_angular,
        'goal': r_goal
    }
    
    return reward, current_dist, reward_components

def reached_goal(env: environment.WalkerEnv) -> bool:
    joint_pos = env.robot.get_joint_world_pos()
    door_x, door_w, door_h = env.door
    dx = np.abs(joint_pos[0] - door_x) - door_w * 0.5
    dy = joint_pos[1] - door_h if joint_pos[1] > door_h else 0
    return dx < 0 and dy == 0

def plot_progress(ep, all_ep_returns, all_ep_success, plots_dir):
    """Plot training progress: episode returns and success rate."""
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    ax.clear()
    ax2.clear()

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return", color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)
    
    ax.plot(all_ep_returns, label="Episode return", color='blue', alpha=0.7)
    
    ax2.set_ylabel("Success Rate", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([-0.05, 1.05])
    
    window = 100
    running_average_success = []
    for i in range(len(all_ep_success)):
        if i < window:
            running_average_success.append(np.mean(all_ep_success[:i+1]))
        else:
            running_average_success.append(np.mean(all_ep_success[i-window+1:i+1]))
    ax2.plot(running_average_success, label="Success rate (100-ep MA)", color='green', linewidth=2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    avg_return = np.mean(all_ep_returns) if all_ep_returns else 0.0
    ax.set_title(f"PPO Episode {ep}, avg_return={avg_return:.1f}")
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, f"episode_{ep:06d}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close(fig)


class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO with squashed Gaussian policy."""
    def __init__(self, obs_dim, action_dim=1):
        super().__init__()
        self.action_dim = action_dim
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.actor_mu = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.obs_dim = obs_dim
        self.number_of_episodes = 0
        
    def forward(self, x):
        features = self.shared(x)
        mu = self.actor_mu(features)  # Unconstrained in R
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        value = self.critic(features)
        return mu, std, value
    
    def get_action_and_value(self, x, squashed_action=None):
        """Get squashed action, log_prob (with tanh correction), entropy, and value.
        
        Args:
            x: Observation tensor
            squashed_action: If provided, compute log_prob for this action (for PPO update).
                             Should be the tanh-squashed action in [-1, 1].
        
        Returns:
            action: Squashed action in [-1, 1] (use this to step env)
            log_prob: Log probability with tanh Jacobian correction
            entropy: Entropy of the base Gaussian (approximate)
            value: Value estimate
        """
        mu, std, value = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        
        if squashed_action is None:
            # Sample u from N(mu, std) using reparameterization
            u = dist.rsample()
            # Squash with tanh to get action in [-1, 1]
            action = torch.tanh(u)
        else:
            # Recover u from stored squashed action using atanh
            # Clamp to avoid numerical issues at boundaries
            action = squashed_action
            u = torch.atanh(action.clamp(-0.999, 0.999))
        
        # Compute log_prob with tanh Jacobian correction
        # log_prob = log_prob_u - sum(log(1 - tanh(u)^2))
        log_prob_u = dist.log_prob(u).sum(dim=-1)
        # Jacobian correction: log(1 - tanh(u)^2) = log(1 - action^2)
        log_jacobian = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        log_prob = log_prob_u - log_jacobian
        
        # Entropy of base Gaussian (approximate, doesn't account for squashing)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)
    def save(self, path: str):    
        torch.save({
            "policy": self.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "number_of_episodes": self.number_of_episodes,
        }, path)

def load_policy(path: str) -> ActorCritic:
    """Load policy from saved checkpoint."""
    data = torch.load(path)
    action_dim = data.get("action_dim", 1)
    policy = ActorCritic(data["obs_dim"], action_dim)
    policy.load_state_dict(data["policy"])
    if "number_of_episodes" in data:
        policy.number_of_episodes = data["number_of_episodes"]
    return policy

def evaluate(env: environment.WalkerEnv, policy: ActorCritic, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> float:
    """Evaluate policy on environment (single step). Returns action in [-1, 1]."""
    obs = get_obs(env)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action, _, _, _ = policy.get_action_and_value(obs_t)
        return float(action.item())

def compute_gae(rewards, values, terminated, truncated, next_value=0, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    if len(rewards) == 0:
        return []
    
    if not (len(rewards) == len(values) == len(terminated) == len(truncated)):
        raise ValueError(f"Length mismatch: rewards={len(rewards)}, values={len(values)}, "
                        f"terminated={len(terminated)}, truncated={len(truncated)}")
    
    advantages = []
    gae = 0
    if len(truncated) > 0 and truncated[-1] and len(terminated) > 0 and not terminated[-1]:
        values = list(values) + [next_value]
    else:
        values = list(values) + [0]
    
    for t in reversed(range(len(rewards))):
        if terminated[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    return advantages


def train(env: environment.WalkerEnv, policy: ActorCritic = None, 
          episodes: int = 2000, gamma: float = 0.99, lr: float = 3e-4, 
          log_dir: str = "runs", plot: bool = True, 
          target_steps: int = 4096, minibatch_size: int = 256, ppo_epochs: int = 4, clip_eps: float = 0.2, 
          vf_coef: float = 0.5, ent_coef: float = 0.01,
          gae_lambda: float = 0.95, episode_save_interval: int = 20, plot_interval: int = 100,
          adaptive_lr: bool = True):
    """
    Train policy using PPO (Proximal Policy Optimization) with timestep-based collection and minibatch updates.
    
    Args:
        episodes: Total number of episodes to train (training continues until this many episodes are completed).
        gamma: Discount factor.
        lr: Learning rate.
        log_dir: Directory to save logs and plots.
        plot: Whether to save plots.
        target_steps: Target number of timesteps to collect before each PPO update (e.g., 2k-16k).
        minibatch_size: Size of minibatches for PPO updates (e.g., 256-2048).
        ppo_epochs: Number of epochs to update on each collected batch.
        clip_eps: PPO clipping parameter.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy bonus coefficient.
        gae_lambda: GAE lambda parameter.
        episode_save_interval: Episodes save interval.
        plot_interval: Episodes interval to save plots.
        adaptive_lr: If True, automatically reduce learning rate by 2x when success rate exceeds 35% and 75%.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    tb_log_dir = os.path.join(log_dir, "tb")
    writer = SummaryWriter(tb_log_dir)
    
    episodes_dir = os.path.join(log_dir, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)
    
    plots_dir = None
    if plot:
        plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    env.reset()
    obs_dim = get_obs(env).shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"Training started at: {start_time_str}")
    print(f"Using device: {device}")
    if policy is None:
        policy = ActorCritic(obs_dim).to(device)
        start_ep = 0
    else:
        policy = policy.to(device)
        start_ep = policy.number_of_episodes
    opt = optim.Adam(policy.parameters(), lr=lr)
    
    lr_reduced_30 = False
    lr_reduced_70 = False

    ep = 0
    total_steps = 0
    all_ep_returns = []
    all_ep_success = []
    temp_policy_file = "temp_policy.pth"

    while ep < episodes:
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_values = []
        batch_terminated = []
        batch_truncated = []
        batch_next_values = []
        batch_replay_steps = []
        batch_ep_returns = []
        batch_ep_success = []
        batch_reward_components = []
        episode_boundaries = []
        
        steps_collected = 0
        num_eps_collected = 0
        
        while steps_collected < target_steps and ep < episodes:
            episode_start_idx = len(batch_obs)
            episode_boundaries.append(episode_start_idx)
            
            env.reset()
            obs = get_obs(env)
            prev_dist = get_distance_to_door(env)
            ep_rewards = []
            replay_steps = []

            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    action, log_prob, _, value = policy.get_action_and_value(obs_t)
                
                action_np = action.cpu().numpy().flatten()
                action_val = float(action_np[0])
                
                env.step(action_val)
                reward, prev_dist, reward_components = calculate_reward(env, prev_dist)
                terminated = reached_goal(env)
                truncated = env.time_count > env.time_limit
                done = terminated or truncated
                next_obs = get_obs(env)
                
                next_value = 0.0
                if truncated and not terminated:
                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        _, _, _, next_val = policy.get_action_and_value(next_obs_t)
                    next_value = next_val.item()

                batch_obs.append(obs)
                batch_actions.append(action_np)
                batch_log_probs.append(log_prob.item())
                batch_rewards.append(reward)
                batch_values.append(value.item())
                batch_terminated.append(terminated)
                batch_truncated.append(truncated)
                batch_next_values.append(next_value)
                batch_reward_components.append(reward_components)
                ep_rewards.append(reward)
                replay_steps.append(env.get_replay_step_data(action_val))
                obs = next_obs
                steps_collected += 1

            batch_replay_steps.append(replay_steps)
            ep_ret = float(np.sum(ep_rewards))
            batch_ep_returns.append(ep_ret)
            batch_ep_success.append(1 if reached_goal(env) else 0)
            
            all_ep_returns.append(ep_ret)
            all_ep_success.append(1 if reached_goal(env) else 0)
            num_eps_collected += 1
            ep += 1
            
            current_ep = start_ep + ep - 1
            if current_ep % episode_save_interval == 0:
                cumulative_returns = []
                cumsum = 0.0
                for r in ep_rewards:
                    cumsum += r
                    cumulative_returns.append(cumsum)
                
                if len(cumulative_returns) != len(replay_steps):
                    raise ValueError(f"Mismatch: ep_rewards length ({len(cumulative_returns)}) != replay_steps length ({len(replay_steps)})")
                
                episode_file = os.path.join(episodes_dir, f"ep_{current_ep:06d}.npz")
                np.savez_compressed(
                    episode_file,
                    points=np.array(env.points, dtype=np.float32),
                    door=np.array(env.door, dtype=np.float32),
                    steps=np.array([(
                        s["a"],
                        *[coord for v in s["vertices1_world"] for coord in v],
                        *[coord for v in s["vertices2_world"] for coord in v],
                        s["joint_pos"][0], s["joint_pos"][1],
                        cumulative_returns[i],
                    ) for i, s in enumerate(replay_steps)], dtype=np.float32),
                    ep_return=np.float32(ep_ret),
                )
                print(f"Saved episode {current_ep} to {episode_file}")

        obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(np.stack(batch_actions), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=device)
        
        all_advantages = []
        episode_boundaries.append(len(batch_obs))
        
        for ep_idx in range(len(episode_boundaries) - 1):
            episode_start_idx = episode_boundaries[ep_idx]
            episode_end_idx = episode_boundaries[ep_idx + 1]
            
            ep_rewards = batch_rewards[episode_start_idx:episode_end_idx]
            ep_values = batch_values[episode_start_idx:episode_end_idx]
            ep_terminated = batch_terminated[episode_start_idx:episode_end_idx]
            ep_truncated = batch_truncated[episode_start_idx:episode_end_idx]
            
            last_step_idx = episode_end_idx - 1
            next_value = batch_next_values[last_step_idx] if last_step_idx < len(batch_next_values) else 0.0
            
            ep_advantages = compute_gae(ep_rewards, ep_values, ep_terminated, ep_truncated, next_value, gamma, gae_lambda)
            all_advantages.extend(ep_advantages)

        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=device)
        returns_tensor = advantages_tensor + torch.tensor(batch_values, dtype=torch.float32, device=device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        num_samples = len(batch_obs)
        indices = np.arange(num_samples)
        
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_losses = []
        epoch_kl_divs = []
        epoch_clip_fractions = []
        
        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, minibatch_size):
                end_idx = min(start_idx + minibatch_size, num_samples)
                mb_indices = indices[start_idx:end_idx]
                
                mb_obs = obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                
                _, new_log_probs, entropy, new_values = policy.get_action_and_value(mb_obs, mb_actions)
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                kl_div = (mb_old_log_probs - new_log_probs).mean()
                clip_fraction = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean()
                value_loss = ((new_values - mb_returns) ** 2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                
                if epoch == ppo_epochs - 1:
                    epoch_policy_losses.append(policy_loss.item())
                    epoch_value_losses.append(value_loss.item())
                    epoch_entropy_losses.append(entropy_loss.item())
                    epoch_kl_divs.append(kl_div.item())
                    epoch_clip_fractions.append(clip_fraction.item())
                
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
        
        total_steps += len(batch_obs)
        
        # Log to TensorBoard
        if len(batch_ep_returns) > 0:
            current_episode = start_ep + ep - 1
            
            # Log training losses (essential metrics)
            if len(epoch_policy_losses) > 0:
                writer.add_scalar("Train/Policy_Loss", np.mean(epoch_policy_losses), current_episode)
                writer.add_scalar("Train/Value_Loss", np.mean(epoch_value_losses), current_episode)
                writer.add_scalar("Train/Entropy_Loss", np.mean(epoch_entropy_losses), current_episode)
                writer.add_scalar("Train/Total_Loss", 
                                 np.mean(epoch_policy_losses) + vf_coef * np.mean(epoch_value_losses) + 
                                 ent_coef * np.mean(epoch_entropy_losses), current_episode)
                
                # Log PPO-specific metrics
                writer.add_scalar("Train/Approx_KL", np.mean(epoch_kl_divs), current_episode)
                writer.add_scalar("Train/Clip_Fraction", np.mean(epoch_clip_fractions), current_episode)
            
            # Log value function metrics
            returns_tensor_np = returns_tensor.cpu().numpy()
            values_tensor_np = np.array(batch_values)  # batch_values is already a list of floats
            mean_return = np.mean(returns_tensor_np)
            mean_value = np.mean(values_tensor_np)
            writer.add_scalar("Train/Mean_Return", mean_return, current_episode)
            writer.add_scalar("Train/Mean_Value", mean_value, current_episode)
            
            # Explained variance: 1 - Var(returns - values) / Var(returns)
            # Measures how well value function predicts returns
            var_returns = np.var(returns_tensor_np)
            if var_returns > 1e-8:
                explained_var = 1.0 - np.var(returns_tensor_np - values_tensor_np) / var_returns
            else:
                explained_var = 0.0
            writer.add_scalar("Train/Explained_Variance", explained_var, current_episode)
            
            # Log reward components (average across all steps in batch)
            if len(batch_reward_components) > 0:
                mean_progress = np.mean([rc['progress'] for rc in batch_reward_components])
                mean_time = np.mean([rc['time'] for rc in batch_reward_components])
                mean_angular = np.mean([rc['angular'] for rc in batch_reward_components])
                mean_goal = np.mean([rc['goal'] for rc in batch_reward_components])
                writer.add_scalar("Reward/Progress", mean_progress, current_episode)
                writer.add_scalar("Reward/Time_Penalty", mean_time, current_episode)
                writer.add_scalar("Reward/Angular_Penalty", mean_angular, current_episode)
                writer.add_scalar("Reward/Goal_Bonus", mean_goal, current_episode)
            
            # Log total steps
            writer.add_scalar("Train/Total_Steps", total_steps, current_episode)
        
        # Log running statistics with 100-episode window
        if len(all_ep_returns) > 0:
            current_episode = start_ep + ep - 1
            
            if len(all_ep_returns) >= 100:
                running_avg_return_100ep = np.mean(all_ep_returns[-100:])
                running_avg_success_100ep = np.mean(all_ep_success[-100:])
                writer.add_scalar("Train/Running_Avg_Return_100ep", running_avg_return_100ep, current_episode)
                writer.add_scalar("Train/Running_Avg_Success_100ep", running_avg_success_100ep, current_episode)
                
                # Adaptive learning rate reduction based on success rate (if enabled)
                if adaptive_lr:
                    current_lr = opt.param_groups[0]['lr']
                    
                    # First threshold: 30-40% success rate (use 35% as middle of range)
                    if not lr_reduced_30 and running_avg_success_100ep >= 0.35:
                        new_lr = current_lr / 2.0
                        for param_group in opt.param_groups:
                            param_group['lr'] = new_lr
                        lr_reduced_30 = True
                        print(f"\n*** LR reduced from {current_lr:.6f} to {new_lr:.6f} (success rate {running_avg_success_100ep:.2%} >= 35%) ***")
                    
                    # Second threshold: 70-80% success rate (use 75% as middle of range)
                    elif not lr_reduced_70 and running_avg_success_100ep >= 0.75:
                        new_lr = current_lr / 2.0
                        for param_group in opt.param_groups:
                            param_group['lr'] = new_lr
                        lr_reduced_70 = True
                        print(f"\n*** LR reduced from {current_lr:.6f} to {new_lr:.6f} (success rate {running_avg_success_100ep:.2%} >= 75%) ***")
                
                # Log learning rate (always log, regardless of adaptive_lr setting)
                writer.add_scalar("Train/Learning_Rate", opt.param_groups[0]['lr'], current_episode)
        
        # Print progress
        if ep % 10 == 0 or len(batch_ep_returns) > 0:
            elapsed_time = time.time() - start_time
            elapsed_minutes = int(elapsed_time // 60)
            avg_return = np.mean(batch_ep_returns) if len(batch_ep_returns) > 0 else 0.0
            avg_success = np.mean(batch_ep_success) if len(batch_ep_success) > 0 else 0.0

            print(f"ep {start_ep + ep - num_eps_collected:4d}-{start_ep + ep - 1:4d} | elapsed: {elapsed_minutes:3d}m | "
                  f"total_steps: {total_steps} | steps_collected: {steps_collected} | "
                  f"avg return {avg_return:8.2f} | avg success {avg_success:.2f}")

        # Save plot as PNG every plot_interval episodes and last episode
        last_finished_ep = start_ep + ep - 1
        # Check if we've crossed a plot_interval boundary or if it's the last episode
        should_save_plot = False
        if plot:
            # Check if any episode in this batch is a multiple of plot_interval
            for i in range(num_eps_collected):
                ep_num = start_ep + ep - num_eps_collected + i
                if ep_num % plot_interval == 0:
                    should_save_plot = True
                    break
            # Or if it's the last episode
            if last_finished_ep == start_ep + episodes - 1:
                should_save_plot = True
        
        if should_save_plot:
            plot_progress(last_finished_ep, all_ep_returns, all_ep_success, plots_dir)
            policy.save(temp_policy_file)
            print(f"Saved policy to temp file: {temp_policy_file} at episode {last_finished_ep}")
    policy.number_of_episodes += episodes
    
    writer.close()
    print(f"TensorBoard logs saved to {tb_log_dir}")
    print(f"View with: tensorboard --logdir {tb_log_dir}")
    
    return policy

