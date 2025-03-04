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
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import os
import sys
import wandb
import yaml
import json
import matplotlib.pyplot as plt
import random
# Import custom walker environment
from env.custom_walker import *
counter = 0

def increment_counter():
    global counter  # Declare that we want to use the global variable
    counter += 1
    print(f"Counter incremented to: {counter}")

best_model = {
    "delta_thigh_mass": 0.0,
    "delta_leg_mass": 0.0,
    "delta_foot_mass": 0.0,
    "source_to_source_mean": 0.00,
    "source_to_source_std": 0.00,
    "source_to_target_mean": 0.00,
    "source_to_target_std": 0.00,
    "#run" : 0
}
best_model_thigh = {
    "delta_thigh_mass": 0.0,
    "delta_leg_mass": 0.0,
    "delta_foot_mass": 0.0,
    "source_to_source_mean": 0.00,
    "source_to_source_std": 0.00,
    "source_to_target_mean": 0.00,
    "source_to_target_std": 0.00,
    "#run" : 0
}
best_model_foot = {
    "delta_thigh_mass": 0.0,
    "delta_leg_mass": 0.0,
    "delta_foot_mass": 0.0,
    "source_to_source_mean": 0.00,
    "source_to_source_std": 0.00,
    "source_to_target_mean": 0.00,
    "source_to_target_std": 0.00,
    "#run" : 0
}
best_model_leg = {
    "delta_thigh_mass": 0.0,
    "delta_leg_mass": 0.0,
    "delta_foot_mass": 0.0,
    "source_to_source_mean": 0.00,
    "source_to_source_std": 0.00,
    "source_to_target_mean": 0.00,
    "source_to_target_std": 0.00,
    "#run" : 0
}

wandb.login()

'''
"learning_rate": 0.0004329019264083943,
    "clip_range": 0.32027045597784387,
    "entropy_coef": 0.0,
    "source_to_source_mean": 1775.9864925,
    "source_to_source_std": 131.99253683218512,
    "source_to_target_mean": 952.5964389999999,
    "source_to_target_std": 25.640810986276435,
    "#run": 5,
    "entropy_coefficient": 0.012119784150808024
'''


LEARNINING_RATE = 0.0004935179450844357
ENTROPY_COEFF = 0.013777855407669348
CLIP_RANGE = 0.265745734677374


# Initialize environments
env_source = Monitor(gym.make("CustomWalker2d-source-v0"))
running_config = ''


model = env_source.sim.model
body_names = model.body_names

right_thigh_index = body_names.index('thigh')
left_thigh_index = body_names.index('thigh_left')
right_foot_index = body_names.index('foot')
right_leg_index = body_names.index('leg')
left_foot_index = body_names.index('foot_left')
left_leg_index = body_names.index('leg_left')


RIGHT_THIGH_MEAN_MASS = env_source.sim.model.body_mass[right_thigh_index]
RIGHT_LEG_MEAN_MASS = env_source.sim.model.body_mass[right_leg_index]
RIGHT_FOOT_MEAN_MASS = env_source.sim.model.body_mass[right_foot_index]
LEFT_THIGH_MEAN_MASS = env_source.sim.model.body_mass[left_thigh_index]
LEFT_LEG_MEAN_MASS = env_source.sim.model.body_mass[left_leg_index]
LEFT_FOOT_MEAN_MASS = env_source.sim.model.body_mass[left_foot_index]

'''print('-'*50)
print('thigh_mean_mass:', THIGH_MEAN_MASS)
print('leg_mean_mass:', LEG_MEAN_MASS)
print('foot_mean_mass:', FOOT_MEAN_MASS)
print('-'*50)
'''
class DomainRandomizationCallback(BaseCallback):
    """Callback for domain randomization, sampling different masses at each episode"""
    def __init__(self, env, verbose=0):
        super(DomainRandomizationCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:

        if any(self.locals['dones']): #Check if the current episode is done

            distribution = args.distribution

            self.env.set_random_parameters(
                RIGHT_THIGH_MEAN_MASS, 
                RIGHT_LEG_MEAN_MASS, 
                RIGHT_FOOT_MEAN_MASS, 
                LEFT_THIGH_MEAN_MASS, 
                LEFT_LEG_MEAN_MASS, 
                LEFT_FOOT_MEAN_MASS, 
                distribution_type = distribution,
                delta_thigh = float(wandb.config.delta_thigh_mass), 
                delta_leg = float(wandb.config.delta_leg_mass), 
                delta_foot = float(wandb.config.delta_foot_mass)
            )
            
        return True
    

def save_best_model(config, mean_ss, std_ss, mean_st, std_st, run_number):
    global running_config
    if running_config == 'config_leg':
        if best_model_leg["source_to_source_mean"] < mean_ss:
            best_model_leg["delta_thigh_mass"] = config.delta_thigh_mass
            best_model_leg["delta_leg_mass"] = config.delta_leg_mass
            best_model_leg["delta_foot_mass"] = config.delta_foot_mass
            best_model_leg["source_to_source_mean"] = mean_ss
            best_model_leg["source_to_source_std"] = std_ss
            best_model_leg["source_to_target_mean"] = mean_st
            best_model_leg["source_to_target_std"] = std_st
            best_model_leg["#run"] = run_number
            # Save the dictionary to a JSON file
            file_path = "./best_s2s_model_between_runs_leg.json"
            with open(file_path, "w") as json_file:
                json.dump(best_model_leg, json_file, indent=4)  

    elif running_config == 'config_foot':
        if best_model_foot["source_to_source_mean"] < mean_ss:
            best_model_foot["delta_thigh_mass"] = config.delta_thigh_mass
            best_model_foot["delta_leg_mass"] = config.delta_leg_mass
            best_model_foot["delta_foot_mass"] = config.delta_foot_mass
            best_model_foot["source_to_source_mean"] = mean_ss
            best_model_foot["source_to_source_std"] = std_ss
            best_model_foot["source_to_target_mean"] = mean_st
            best_model_foot["source_to_target_std"] = std_st
            best_model_foot["#run"] = run_number
            # Save the dictionary to a JSON file
            file_path = "./best_s2s_model_between_runs_foot.json"
            with open(file_path, "w") as json_file:
                json.dump(best_model_foot, json_file, indent=4)

    elif running_config == 'config_thigh':
        if best_model_thigh["source_to_source_mean"] < mean_ss:
            best_model_thigh["delta_thigh_mass"] = config.delta_thigh_mass
            best_model_thigh["delta_leg_mass"] = config.delta_leg_mass
            best_model_thigh["delta_foot_mass"] = config.delta_foot_mass
            best_model_thigh["source_to_source_mean"] = mean_ss
            best_model_thigh["source_to_source_std"] = std_ss
            best_model_thigh["source_to_target_mean"] = mean_st
            best_model_thigh["source_to_target_std"] = std_st
            best_model_thigh["#run"] = run_number
            # Save the dictionary to a JSON file
            file_path = "./best_s2s_model_between_runs_thigh.json"
            with open(file_path, "w") as json_file:
                json.dump(best_model_thigh, json_file, indent=4)
                
    else: return

# -----------------------------------------------#
# --------------- CONFIGURATIONS ----------------#
# -----------------------------------------------#

# UNIFORM CONFIGURATIONS
sweep_config_foot = {
        "method": "grid",
        "name": "sweep",
        "metric":
            {"name": "source_to_source_mean", "goal": "maximize"},
        "parameters": {
            "delta_thigh_mass":{"values":[0]},
            "delta_leg_mass":{"values":[0]},
            "delta_foot_mass":{"values":[0.30, 0.35, 0.40]}
        }
}
sweep_config_leg = {
        "method": "grid",
        "name": "sweep",
        "metric":
            {"name": "source_to_source_mean", "goal": "maximize"},
        "parameters": {
            "delta_thigh_mass":{"values":[0]},
            "delta_leg_mass":{"values":[0.30, 0.35, 0.40]},
            "delta_foot_mass":{"values":[0]}
        }
}
sweep_config_thigh = {
        "method": "grid",
        "name": "sweep",
        "metric":
            {"name": "source_to_source_mean", "goal": "maximize"},
        "parameters": {
            "delta_thigh_mass":{"values":[0.30, 0.35, 0.40]},
            "delta_leg_mass":{"values":[0]},
            "delta_foot_mass":{"values":[0]}
        }
}
final_sweep_config = {
        "method": "grid",
        "name": "sweep",
        "metric":
            {"name": "source_to_source_mean", "goal": "maximize"},
        "parameters": {
            "delta_thigh_mass":{"values":[0.35]},
            "delta_leg_mass":{"values":[0.35]},
            "delta_foot_mass":{"values":[0.35]}
        }
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
    domain_randomization_callback = DomainRandomizationCallback(env)
    model.learn(total_timesteps=timesteps, callback=[eval_callback, domain_randomization_callback], progress_bar=True)
    best_model_path = os.path.join(model_save_path+"run_"+str(counter), "best_model.zip")
    increment_counter()
    if os.path.exists(best_model_path):
        best_model = PPO.load(best_model_path)
        print("Best model loaded for evaluation.")
        return best_model
    print("No best model found, returning the final trained model.")
    return model


def sweep_train():

    # Initialize WandB project
    run_number = counter
    os.makedirs("./logs_train/run_"+str(run_number), exist_ok=True)
    run = wandb.init(project="custom-walker-ppo", dir="./logs_train/run_"+str(run_number))
    config = run.config

    # Initialize environments
    env_source = Monitor(gym.make("CustomWalker2d-source-v0"))
    env_target = Monitor(gym.make("CustomWalker2d-target-v0"))

    print(config)
    # Train the source model
    source_model_path = "./"
    os.makedirs(source_model_path, exist_ok=True)
    model_source = train_policy(env_source, source_model_path, args.timesteps, LEARNINING_RATE, CLIP_RANGE, ENTROPY_COEFF)

    # Evaluate Source Model

    # Source to Source
    mean_ss, std_ss = evaluate_policy(model_source, env_source, n_eval_episodes=args.eval_episodes, render=args.render)
    print(f"Source to Source: Mean reward = {mean_ss}, Std reward = {std_ss}")

    # Source to Target
    mean_st, std_st = evaluate_policy(model_source, env_target, n_eval_episodes=args.eval_episodes, render=args.render)
    print(f"Source to Target: Mean reward = {mean_st}, Std reward = {std_st}")

    save_best_model(config, mean_ss, std_ss, mean_st, std_st, run_number)
       

    if running_config != 'config_leg' and running_config != 'config_foot' and running_config != 'config_thigh':
        if best_model["source_to_source_mean"] < mean_ss:
            best_model["delta_thigh_mass"] = config.delta_thigh_mass
            best_model["delta_leg_mass"] = config.delta_leg_mass
            best_model["delta_foot_mass"] = config.delta_foot_mass
            best_model["source_to_source_mean"] = mean_ss
            best_model["source_to_source_std"] = std_ss
            best_model["source_to_target_mean"] = mean_st
            best_model["source_to_target_std"] = std_st
            best_model["#run"] = run_number
            # Save the dictionary to a JSON file
            file_path = "./best_s2s_model_between_runs.json"
            with open(file_path, "w") as json_file:
                json.dump(best_model, json_file, indent=4)


    # Log results for source model
    wandb.log({
        "delta_thigh_mass": config.delta_thigh_mass,
        "delta_leg_mass": config.delta_leg_mass,
        "delta_foot_mass": config.delta_foot_mass,
        "source_to_source_mean": mean_ss,
        "source_to_source_std": std_ss,
        "source_to_target_mean": mean_st,
        "source_to_target_std": std_st,
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
    env_source = Monitor(gym.make("CustomWalker2d-source-v0"))
    env_target = Monitor(gym.make("CustomWalker2d-target-v0"))
    global running_config

    # Training and testing directories
    model_path = "./models"
    os.makedirs(model_path, exist_ok=True)

    if not args.test:
        print("Training policies...")

        # Load config for source models, then train

        # Optimization for source --> source 

        # Different deltas for foot mass
        """ sweep_id_source = wandb.sweep(sweep_config_foot, project="custom-walker-ppo")
        running_config = 'config_foot'
        print(f"Running config: {running_config}")
        wandb.agent(sweep_id_source, function=sweep_train)

        # Different deltas for leg mass
        sweep_id_source = wandb.sweep(sweep_config_leg, project="custom-walker-ppo")
        running_config = 'config_leg'
        print(f"Running config: {running_config}")

        wandb.agent(sweep_id_source, function=sweep_train)
        
        # Different deltas for thigh mass
        sweep_id_source = wandb.sweep(sweep_config_thigh, project="custom-walker-ppo")
        running_config = 'config_thigh'
        print(f"Running config: {running_config}")
        wandb.agent(sweep_id_source, function=sweep_train) """
        
        # Combination of best models
        sweep_id_source = wandb.sweep(final_sweep_config, project="custom-walker-ppo")
        running_config = 'config_final'
        wandb.agent(sweep_id_source, function=sweep_train)

    else:
        print("Testing trained policies...")
        print(os.path.join(model_path, args.model))
        # Load trained models
        model = PPO.load(os.path.join(model_path, args.model))
        
        env_source_id = "CustomWalker2d-source-v0"   # env_id name
        env_target_id = "CustomWalker2d-target-v0"

        # Create a list of environments with different seeds
        env_fns_source = [make_env(env_source_id, i) for i in range(args.eval_episodes)]
        vec_env_source = DummyVecEnv(env_fns_source)  # creates the vector of environments
        env_fns_target = [make_env(env_target_id, i + random.randint(0, 100)) for i in range(args.eval_episodes)]
        vec_env_target = DummyVecEnv(env_fns_target)  # creates the vector of environments

        # Evaluate policies
        print("Evaluating policies...")

        if args.plot:
            episode_rewards, episode_lengths = evaluate_policy(model, env_target, n_eval_episodes=args.eval_episodes, render=args.render, return_episode_rewards=True)
            
            if args.distribution == 'uniform':
                np.save("../../Plots/evaluation/walker/udr_3_delta.npy", episode_rewards)
            elif args.distribution == 'gaussian':
                np.save("../Plots/evaluation/walker/extension_gaussian_rewards.npy", episode_rewards)

            # Plot the rewards per episode
            plt.figure(figsize=(12, 6))
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
            wandb.init(project="custom-walker-ppo")
            config = wandb.config
            
            # Source to Source
            mean_ss, std_ss = evaluate_policy(model, vec_env_source, n_eval_episodes=args.eval_episodes, render=args.render)
            print(f"Source to Source: Mean reward = {mean_ss}, Std reward = {std_ss}")
            
            # Source to Target
            mean_st, std_st = evaluate_policy(model, vec_env_target, n_eval_episodes=args.eval_episodes, render=args.render)
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
    parser.add_argument('--timesteps', type=int, default=500000, help='Number of training timesteps')
    parser.add_argument('--test', action='store_true', help='Test the trained models')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--model', type=str, default="best_model", help="Model to use for testing")
    parser.add_argument('--plot', action='store_true', help='Plot the tested models evaluation results')
    parser.add_argument('--distribution', type=str, default="uniform", help="Uniform to use during trainig")
    parser.add_argument('--box_dim', action='store_true', help="Print body parts")
    return parser.parse_args(args)


if __name__ == '__main__':
    
    #args = parse_args(['--box_dim'])

    # training
    #args = parse_args(['--timesteps', '700000'])

    #args = parse_args(['--timesteps', '1000000', '--distribution', 'uniform'])
    
    # testing
    args = parse_args(['--test', '--eval_episodes', '50', '--model', '3_delta/best_model', '--distribution', 'uniform', '--plot'])
    
    #args = parse_args(['--test', '--render', '--model', 'UDR gaussian/3_delta/best_model'])

    main(args)
    
    