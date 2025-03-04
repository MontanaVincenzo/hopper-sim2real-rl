"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
""" import gym
from env.custom_hopper import *

def main():
    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    
        TODO:

            - train a policy with stable-baselines3 on the source Hopper env
            - test the policy with stable-baselines3 on <source,target> Hopper envs (hint: see the evaluate_policy method of stable-baselines3)
    

if __name__ == '__main__':
    main() """


import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import argparse
import os
import sys
import wandb
import yaml

# Import custom Hopper environment
from env.custom_hopper import *


# -----------------------------------------------#
# --------------- CONFIGURATIONS ----------------#
# -----------------------------------------------#
sweep_config_source_s2s_optimized = {
    "method": "random",  # Random search
    "metric": {
        "name": "source_to_source_mean",  # Optimize for source-to-target mean reward
        "goal": "maximize",
    },
    "parameters": {
        "learning_rate": {
            "min": 1e-5,  # Lower minimum for fine-tuning
            "max": 1e-3,  # Slightly higher maximum to allow more exploration
        },
        "clip_range": {
            "min": 0.25,  # Narrowed range for better stability
            "max": 0.35,  # Centered around commonly effective values
        },
        "ent_coef": {
            "min": 1e-3,  # Encourages more entropy exploration
            "max": 0.02,  # Limits the entropy penalty for stability
        }
    },
}

sweep_config_source_s2t_optimized = {
    "method": "random",
    "metric": {
        "name": "source_to_target_mean",  # Optimize for source-to-target mean reward
        "goal": "maximize",
    },
    "parameters": {
        "learning_rate": {
            "min": 1e-5,
            "max": 1e-3,
        },
        "clip_range": {
            "min": 0.25,
            "max": 0.35,
        },
        "ent_coef": {
            "min": 1e-3,
            "max": 0.02,
        }
    },
}
sweep_config_target = {
    "method": "random",
    "metric": {
        "name": "target_to_target_mean",  # Optimize for target-to-target mean reward
        "goal": "maximize",
    },
    "parameters": {
        "learning_rate": {
            "min": 1e-5,
            "max": 1e-3,
        },
        "clip_range": {
            "min": 0.25,
            "max": 0.35,
        },
        "ent_coef": {
            "min": 1e-3,
            "max": 0.02,
        }
    },
}


# -----------------------------------------------#
# ------------------- TRAIN ---------------------#
# -----------------------------------------------#
def train_policy(env, model_save_path, log_path, timesteps, learning_rate=0.0003, clip_range=0.2, ent_coef=0.01):
    """Train a PPO policy on the given environment."""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1
    )
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_save_path,
        log_path=log_path,
        eval_freq=10000,
        n_eval_episodes=10
    )
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
    return model


def sweep_train_source(optimization_env='source'):

    # Initialize WandB project
    wandb.init(project="custom-hopper-ppo")
    config = wandb.config

    if optimization_env == 'source':
            
        """Sweep for optimizing the source model (source_to_target_mean)."""

        # Initialize environments
        env_source = Monitor(gym.make("CustomHopper-source-v0"))
        env_target = Monitor(gym.make("CustomHopper-target-v0"))

        # Train the source model
        source_model_path = "./models/source/s2s_optimized"
        os.makedirs(source_model_path, exist_ok=True)
        model_source = train_policy(env_source, source_model_path, "./logs/source/s2s_optimized", args.timesteps, config.learning_rate, config.clip_range, config.ent_coef)

        # Evaluate Source Model

        # Source to Source
        mean_ss, std_ss = evaluate_policy(model_source, env_source, n_eval_episodes=args.eval_episodes, render=args.render)
        print(f"Source to Source: Mean reward = {mean_ss}, Std reward = {std_ss}")

        # Source to Target
        mean_st, std_st = evaluate_policy(model_source, env_target, n_eval_episodes=args.eval_episodes, render=args.render)
        print(f"Source to Target: Mean reward = {mean_st}, Std reward = {std_st}")

        # Log results for source model
        wandb.log({
            "source_to_source_mean": mean_ss,
            "source_to_source_std": std_ss,
            "source_to_target_mean": mean_st,
            "source_to_target_std": std_st,
            "learning_rate": config.learning_rate,
            "clip_range": config.clip_range,
            "ent_coef": config.ent_coef,
            "timesteps": args.timesteps,
        })
    elif optimization_env == 'target':   
        """Sweep for optimizing the source model (source_to_target_mean)."""
        

        # Initialize environments
        env_source = Monitor(gym.make("CustomHopper-source-v0"))
        env_target = Monitor(gym.make("CustomHopper-target-v0"))

        # Train the source model
        source_model_path = "./models/source/s2t_optimized"
        os.makedirs(source_model_path, exist_ok=True)
        model_source = train_policy(env_source, source_model_path, "./logs/source/s2t_optimized", args.timesteps, config.learning_rate, config.clip_range, config.ent_coef)

        # Evaluate Source Model

        # Source to Source
        mean_ss, std_ss = evaluate_policy(model_source, env_source, n_eval_episodes=args.eval_episodes, render=args.render)
        print(f"Source to Source: Mean reward = {mean_ss}, Std reward = {std_ss}")

        # Source to Target
        mean_st, std_st = evaluate_policy(model_source, env_target, n_eval_episodes=args.eval_episodes, render=args.render)
        print(f"Source to Target: Mean reward = {mean_st}, Std reward = {std_st}")

        # Log results for source model
        wandb.log({
            "source_to_source_mean": mean_ss,
            "source_to_source_std": std_ss,
            "source_to_target_mean": mean_st,
            "source_to_target_std": std_st,
            "learning_rate": config.learning_rate,
            "clip_range": config.clip_range,
            "ent_coef": config.ent_coef,
            "timesteps": args.timesteps,
        })

def sweep_train_target():
    """Sweep for optimizing the target model (target_to_target_mean)."""
    # Initialize WandB project
    wandb.init(project="custom-hopper-ppo")
    config = wandb.config
    print(config)
    # Initialize environments
    env_target = Monitor(gym.make("CustomHopper-target-v0"))

    # Train the target model
    target_model_path = "./models/target/"
    os.makedirs(target_model_path, exist_ok=True)
    model_target = train_policy(env_target, target_model_path, "./logs/target/", args.timesteps, config.learning_rate, config.clip_range, config.ent_coef)
    
    # Evaluate Target Model 
    
    # Target to Target
    mean_tt, std_tt = evaluate_policy(model_target, env_target, n_eval_episodes=args.eval_episodes, render=args.render)
    print(f"Target to Target: Mean reward = {mean_tt}, Std reward = {std_tt}")

    # Log results for target model
    wandb.log({
        "target_to_target_mean": mean_tt,
        "target_to_target_std": std_tt,
        "learning_rate": config.learning_rate,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
        "timesteps": args.timesteps,
    })


# -----------------------------------------------#
# ------------------- MAIN ----------------------#
# -----------------------------------------------#
def main(args):
    
    # Initialize environments
    env_source = Monitor(gym.make("CustomHopper-source-v0"))
    env_target = Monitor(gym.make("CustomHopper-target-v0"))

    # Training and testing directories
    source_model_path_s2s_optimized = "./"
    source_model_path_s2t_optimized = "./"
    target_model_path = "./"
    os.makedirs(source_model_path_s2s_optimized, exist_ok=True)
    os.makedirs(source_model_path_s2t_optimized, exist_ok=True)
    os.makedirs(target_model_path, exist_ok=True)

    if not args.test:
        print("Training policies...")

        # Load config for source models, then train

        # Optimization for source --> source 
        sweep_id_source = wandb.sweep(sweep_config_source_s2s_optimized, project="custom-hopper-ppo")
        wandb.agent(sweep_id_source, function=lambda: sweep_train_source('source'), count=10)

        # Optimization for source --> target 
        sweep_id_source = wandb.sweep(sweep_config_source_s2t_optimized, project="custom-hopper-ppo")
        wandb.agent(sweep_id_source, function=sweep_train_source('target'), count=10)
        
        #Load config for target model, then train
        sweep_id_target = wandb.sweep(sweep_config_target, project="custom-hopper-ppo")
        wandb.agent(sweep_id_target, function=sweep_train_target, count=10)

    else:
        print("Testing trained policies...")
        # Load trained models
        model_source_s2s_optimized = PPO.load(os.path.join(source_model_path_s2s_optimized, "best_model"))
        
        # model_source_s2t_optimized = PPO.load(os.path.join(source_model_path_s2t_optimized, "best_model")) # Probably useless
        
        model_target = PPO.load(os.path.join(target_model_path, "best_model"))

        # Evaluate policies
        print("Evaluating policies...")

        # Source to Source
        mean_ss, std_ss = evaluate_policy(model_source_s2s_optimized, env_source, n_eval_episodes=args.eval_episodes, render=args.render)
        print(f"Source to Source: Mean reward = {mean_ss}, Std reward = {std_ss}")

        # Source to Target
        mean_st, std_st = evaluate_policy(model_source_s2s_optimized, env_target, n_eval_episodes=args.eval_episodes, render=args.render)
        print(f"Source to Target: Mean reward = {mean_st}, Std reward = {std_st}")

        # Target to Target
        mean_tt, std_tt = evaluate_policy(model_target, env_target, n_eval_episodes=args.eval_episodes, render=args.render)
        print(f"Target to Target: Mean reward = {mean_tt}, Std reward = {std_tt}")



# -----------------------------------------------#
# -------------------ARGUMENTS ------------------#
# -----------------------------------------------#

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=300000, help='Number of training timesteps')
    parser.add_argument('--test', action='store_true', help='Test the trained models')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    return parser.parse_args(args)


if __name__ == '__main__':
    
    args = parse_args(['--test', '--render'])

    main(args)
    