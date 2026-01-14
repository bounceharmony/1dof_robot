import environment
import brains


if __name__ == "__main__":
    env = environment.WalkerEnv(width=800, height=600, time_limit=20, substeps=5, 
                                max_torque=120000.0, seed=0, number_of_ground_points=20, fps=30)
    
    # Load existing policy or start fresh
    policy = None  # Set to brains.load_policy("path/to/policy.pth") to resume training
    policy = brains.train(env, policy, episodes=1000, lr=5e-5, plot=True, 
                          target_steps=4096, minibatch_size=256, 
                          episode_save_interval=20, plot_interval=200, ppo_epochs=3,
                          log_dir="runs", adaptive_lr=False)
    policy.save("runs/policy.pth")
