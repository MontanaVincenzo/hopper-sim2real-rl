import matplotlib.pyplot as plt
import re
import sys
import argparse



def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--log_path',  type=str, default='', help='path to the output.log file')
    parser.add_argument('--save_path',  type=str, default='figure_plot.png', help='name of the figure')
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    #log_file_path = "../../PPO/source_logs_train/run_5/wandb/latest-run/files/output.log"

    log_file_path = "../../Extension_Hopper/logs_train/UDR uniform/run_9/wandb/run-20250119_201502-ojob71cw/files/output.log"
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    # Regular expression to extract episode rewards from the log
    reward_pattern = r"episode_reward=([\d\.]+)"

    # Extracting rewards
    rewards = [float(match) for match in re.findall(reward_pattern, log_content)]
    timesteps = [i * 10000 for i in range(1, len(rewards) + 1)]
    # Plotting the rewards
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, linestyle='-', label='Episode Rewards')
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.grid()
    plt.legend()
    if args.save:
        plt.savefig(args.save_path)
    else:
        plt.show()
    