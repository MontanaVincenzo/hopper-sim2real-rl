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
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import os
import sys
import wandb
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import random
# Import custom Hopper environment
from env.custom_hopper import *
counter = 0

def increment_counter():
    global counter  # Declare that we want to use the global variable
    counter += 1
    print(f"Counter incremented to: {counter}")

# -----------------------------------------------#
# --------------- CONFIGURATIONS ----------------#
# -----------------------------------------------#


best_source_model = {
    "learning_rate": 0.0,
    "clip_range": 0.0,
    "entropy_coef": 0.0,
    "source_to_source_mean": 0.00,
    "source_to_source_std": 0.00,
    "source_to_target_mean": 0.00,
    "source_to_target_std": 0.00,
    "#run" : 0
}
best_target_model = {
    "learning_rate": 0.0,
    "clip_range": 0.0,
    "entropy_coef": 0.0,
    "target_to_target_mean": 0.00,
    "target_to_target_std": 0.00,
    "#run" : 0
}
sweep_config_source = {
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
def train_policy(env, model_save_path, timesteps, learning_rate=0.0003, clip_range=0.2, ent_coef=0.01):
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
        best_model_save_path=model_save_path+"run_"+str(counter),
        eval_freq=10000,
        n_eval_episodes=10
    )
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
    # Load the best model found during training
    best_model_path = os.path.join(model_save_path+"run_"+str(counter), "best_model.zip")
    increment_counter()
    if os.path.exists(best_model_path):
        best_model = PPO.load(best_model_path)
        print("Best model loaded for evaluation.")
        return best_model
    print("No best model found, returning the final trained model.")
    return model


def sweep_train_source():

    # Initialize WandB project
    run_number = counter
    os.makedirs("./source_logs_train/run_"+str(run_number), exist_ok=True)
    wandb.init(project="custom-hopper-ppo", dir="./source_logs_train/run_"+str(run_number))
    config = wandb.config

            
    """Sweep for optimizing the source model (source_to_target_mean)."""

    # Initialize environments
    env_source = Monitor(gym.make("CustomHopper-source-v0"))
    env_target = Monitor(gym.make("CustomHopper-target-v0"))

    # Train the source model
    source_model_path = "./models/source/"
    os.makedirs(source_model_path, exist_ok=True)
    model_source = train_policy(env_source, source_model_path, args.timesteps, config.learning_rate, config.clip_range, config.ent_coef)

    # Evaluate Source Model

    # Source to Source
    mean_ss, std_ss = evaluate_policy(model_source, env_source, n_eval_episodes=args.eval_episodes, render=args.render)
    print(f"Source to Source: Mean reward = {mean_ss}, Std reward = {std_ss}")

    # Source to Target
    mean_st, std_st = evaluate_policy(model_source, env_target, n_eval_episodes=args.eval_episodes, render=args.render)
    print(f"Source to Target: Mean reward = {mean_st}, Std reward = {std_st}")

    if best_source_model["source_to_source_mean"] < mean_ss:
        best_source_model["learning_rate"] = config.learning_rate
        best_source_model["clip_range"] = config.clip_range
        best_source_model["entropy_coefficient"] = config.ent_coef
        best_source_model["source_to_source_mean"] = mean_ss
        best_source_model["source_to_source_std"] = std_ss
        best_source_model["source_to_target_mean"] = mean_st
        best_source_model["source_to_target_std"] = std_st
        best_source_model["#run"] = run_number
        # Save the dictionary to a JSON file
        file_path = "./best_source_model_between_runs.json"
        with open(file_path, "w") as json_file:
            json.dump(best_source_model, json_file, indent=4)  # `indent=4` for pretty printing

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
    run_number = counter
    """Sweep for optimizing the target model (target_to_target_mean)."""
    # Initialize WandB project
    os.makedirs("./target_logs_train/run_"+str(run_number), exist_ok=True)
    wandb.init(project="custom-hopper-ppo", dir="./target_logs_train/run_"+str(run_number))
    config = wandb.config
    print(config)
    # Initialize environments
    env_target = Monitor(gym.make("CustomHopper-target-v0"))

    # Train the target model
    target_model_path = "./models/target/"
    os.makedirs(target_model_path, exist_ok=True)
    model_target = train_policy(env_target, target_model_path, args.timesteps, config.learning_rate, config.clip_range, config.ent_coef)
    
    # Evaluate Target Model 
    
    # Target to Target
    mean_tt, std_tt = evaluate_policy(model_target, env_target, n_eval_episodes=args.eval_episodes, render=args.render)
    print(f"Target to Target: Mean reward = {mean_tt}, Std reward = {std_tt}")


    if best_target_model["target_to_target_mean"] < mean_tt:
        best_target_model["learning_rate"] = config.learning_rate
        best_target_model["clip_range"] = config.clip_range
        best_target_model["entropy_coefficient"] = config.ent_coef
        best_target_model["target_to_target_mean"] = mean_tt
        best_target_model["source_to_target_std"] = std_tt
        best_target_model["#run"] = run_number
        # Save the dictionary to a JSON file
        file_path = "./best_target_model_between_runs.json"
        with open(file_path, "w") as json_file:
            json.dump(best_target_model, json_file, indent=4)  # `indent=4` for pretty printing
    
    # Log results for target model
    wandb.log({
        "target_to_target_mean": mean_tt,
        "target_to_target_std": std_tt,
        "learning_rate": config.learning_rate,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
        "timesteps": args.timesteps,
    })

def make_env(env_id, rank, seed=0):
    def _init():
        env = Monitor(gym.make(env_id))
        env.seed(seed + rank)
        return env
    return _init


# -----------------------------------------------#
# ------------------- MAIN ----------------------#
# -----------------------------------------------#
def main(args):
    
    # Initialize environments
    env_source = Monitor(gym.make("CustomHopper-source-v0"))
    env_target = Monitor(gym.make("CustomHopper-target-v0"))

    # Training and testing directories
    source_model_path = "./models/source/run_5" # Ricorda di aggiungere il percorso che vuoi testare
    target_model_path = "./models/target/run_5" # Ricorda di aggiungere il percorso che vuoi testare
    os.makedirs(source_model_path, exist_ok=True)
    os.makedirs(target_model_path, exist_ok=True)

    if not args.test:
        print("Training policies...")

        # Load config for source models, then train

        # Optimization for source --> source 
        # sweep_id_source = wandb.sweep(sweep_config_source, project="custom-hopper-ppo")
        # wandb.agent(sweep_id_source, function=sweep_train_source, count=10)
        
        # Load config for target model, then train
        # sweep_id_target = wandb.sweep(sweep_config_target, project="custom-hopper-ppo")
        # wandb.agent(sweep_id_target, function=sweep_train_target, count=10)

    else:
        print('-'*70)
        
        print("Testing trained policies...")
        # Load trained models
        model_source = PPO.load(os.path.join(source_model_path, "best_model"))
                
        model_target = PPO.load(os.path.join(target_model_path, "best_model"))

        env_id_source = "CustomHopper-source-v0"   # env_id name
        env_id_target = "CustomHopper-target-v0"   # env_id name

        # Create a list of environments with different seeds
        env_fns_source = [make_env(env_id_source, random.randint(0, 1000)) for i in range(args.eval_episodes)]
        vec_env_source = DummyVecEnv(env_fns_source)  # creates the vector of environments
        env_fns_target = [make_env(env_id_target, random.randint(0, 1000)) for i in range(args.eval_episodes)]
        vec_env_target = DummyVecEnv(env_fns_target)  # creates the vector of environments

        # Evaluate policies
        print("Evaluating policies...")

        if args.plot:
            # Source to Target
            episode_rewards, episode_lengths = evaluate_policy(model_source, env_target, n_eval_episodes=args.eval_episodes, render=args.render, return_episode_rewards=True)
            np.save("../Plots/s2t_rewards.npy", episode_rewards)

            # Plot the rewards per episode
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-', label='Episode Reward')
            plt.axhline(y=sum(episode_rewards) / len(episode_rewards), color='r', linestyle='--', label='Mean Reward')

            # Add titles and labels
            plt.title('Rewards per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.xticks(range(1, len(episode_rewards) + 1))
            plt.legend()
            plt.grid()

            # Show the plot
            plt.show()
            
            
            # Target to Target
            episode_rewards, episode_lengths = evaluate_policy(model_target, env_target, n_eval_episodes=args.eval_episodes, render=args.render, return_episode_rewards=True)
            np.save("../Plots/t2t_rewards.npy", episode_rewards)

            # Plot the rewards per episode
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-', label='Episode Reward')
            plt.axhline(y=sum(episode_rewards) / len(episode_rewards), color='r', linestyle='--', label='Mean Reward')

            # Add titles and labels
            plt.title('Rewards per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.xticks(range(1, len(episode_rewards) + 1))
            plt.legend()
            plt.grid()

            # Show the plot
            plt.show()
        
        else: 
            wandb.init(project="custom-hopper-ppo")
            config = wandb.config

            # Source to Source
            mean_ss, std_ss = evaluate_policy(model_source, vec_env_source, n_eval_episodes=args.eval_episodes, render=args.render)
            print(f"Source to Source: Mean reward = {mean_ss}, Std reward = {std_ss}") 

            # Source to Target
            episode_rewards, episode_lengths = evaluate_policy(model_source, vec_env_target, n_eval_episodes=args.eval_episodes, render=args.render, return_episode_rewards=True)
            print(f"Source to Target: Mean reward = {mean_st}, Std reward = {std_st}")

            # Target to Target
            mean_st, std_st = evaluate_policy(model_target, vec_env_target, n_eval_episodes=args.eval_episodes, render=args.render)
            print(f"Source to Target: Mean reward = {mean_st}, Std reward = {std_st}") 

            wandb.log({
                "source_to_source_mean": mean_ss,
                "source_to_source_std": std_ss,
                "source_to_target_mean": mean_st,
                "source_to_target_std": std_st,
                "timesteps": args.timesteps,
            })

# -----------------------------------------------#
# -------------------ARGUMENTS ------------------#
# -----------------------------------------------#

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=300000, help='Number of training timesteps')
    parser.add_argument('--test', action='store_true', help='Test the trained models')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--plot', action='store_true', help='Plot the tested models evaluation results')
    return parser.parse_args(args)


if __name__ == '__main__':
    

    #args = parse_args(['--timesteps', '700000'])

    args = parse_args(['--test', '--plot', '--eval_episodes', '50'])
    

    main(args)
    