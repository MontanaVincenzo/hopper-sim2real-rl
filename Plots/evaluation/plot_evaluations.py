import numpy as np
import matplotlib.pyplot as plt

#EXTENSION_TYPE = 'Hopper'
EXTENSION_TYPE = 'Walker2D'


if __name__ == '__main__':

    if EXTENSION_TYPE == 'Hopper':
        # Load the rewards from files
        udr_rewards = np.load("./hopper/udr_rewards.npy").tolist()
        s2t_rewards = np.load("./hopper/s2t_rewards.npy").tolist()
        t2t_rewards = np.load("./hopper/t2t_rewards.npy").tolist()
        extension_uniform_rewards_3 = np.load("./hopper/extension_uniform_rewards_3_delta.npy").tolist()
        extension_uniform_rewards_thigh = np.load("./hopper/extension_uniform_rewards_thigh_delta.npy").tolist()
        extension_gaussian_rewards_3 = np.load("./hopper/extension_gaussian_rewards_3_delta.npy").tolist()

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot each reward list
        #plt.plot(range(1, len(udr_rewards) + 1), udr_rewards, linestyle='-', label='UDR Rewards', markersize=4, alpha=0.8)
        plt.plot(range(1, len(t2t_rewards) + 1), t2t_rewards, linestyle='-', label='T2T Rewards', markersize=4, alpha=0.8)
        plt.plot(range(1, len(s2t_rewards) + 1), s2t_rewards, linestyle='-', label='S2T Rewards', markersize=4, alpha=0.8)
        #plt.plot(range(1, len(extension_uniform_rewards_3) + 1), extension_uniform_rewards_3, linestyle='-', label='Uniform Rewards 3 Delta', markersize=4, alpha=0.8)
        #plt.plot(range(1, len(extension_uniform_rewards_thigh) + 1), extension_uniform_rewards_thigh, linestyle='-', label='Uniform Rewards Thigh', markersize=4, alpha=0.8)
        plt.plot(range(1, len(extension_gaussian_rewards_3) + 1), extension_gaussian_rewards_3, linestyle='-', label='Gaussian Rewards', markersize=4, alpha=0.8)

        # Add a mean reward line for each list
        #plt.axhline(y=np.mean(udr_rewards), color='r', linestyle='--', label='Mean UDR Rewards')
        plt.axhline(y=np.mean(t2t_rewards), color='g', linestyle='--', label='Mean T2T Rewards')
        plt.axhline(y=np.mean(s2t_rewards), color='cyan', linestyle='--', label='Mean S2T Rewards')
        #plt.axhline(y=np.mean(extension_uniform_rewards_3), color='b', linestyle='--', label='Mean Uniform Rewards')
        #plt.axhline(y=np.mean(extension_uniform_rewards_thigh), color='m', linestyle='--', label='Mean Uniform Rewards Thigh')
        plt.axhline(y=np.mean(extension_gaussian_rewards_3), color='m', linestyle='--', label='Mean Gaussian Rewards')

        # Add titles and labels
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.xticks(range(1, max(len(udr_rewards), len(t2t_rewards), len(extension_uniform_rewards_3)) + 1))

        # Adjust legend location
        plt.legend(loc='center left', fontsize=10, frameon=True, shadow=True, borderpad=1)
        plt.grid(alpha=0.5, linestyle='--')

        # Show the plot
        plt.tight_layout()
        plt.savefig('./hopper/figures/gdr_3_delta')
        plt.show()
    
    if EXTENSION_TYPE == 'Walker2D':
        # Load the rewards from files
        udr_rewards = np.load("./walker/udr_rewards.npy").tolist()
        s2t_rewards = np.load("./walker/s2t_rewards.npy").tolist()
        t2t_rewards = np.load("./walker/t2t_rewards.npy").tolist()
        extension_uniform_rewards_3 = np.load("./walker/udr_3_delta.npy").tolist()
        #extension_uniform_rewards_thigh = np.load("./walker/extension_uniform_rewards_thigh_delta.npy").tolist()
        extension_gaussian_rewards_3 = np.load("./walker/gdr_3_delta_same_sx_dx.npy").tolist()
        extension_gaussian_rewards_thigh = np.load("./walker/gdr_thigh_same_sx_dx.npy").tolist()

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot each reward list
        #plt.plot(range(1, len(udr_rewards) + 1), udr_rewards, linestyle='-', label='UDR Rewards', markersize=4, alpha=0.8)
        plt.plot(range(1, len(t2t_rewards) + 1), t2t_rewards, linestyle='-', label='T2T Rewards', markersize=4, alpha=0.8)
        #plt.plot(range(1, len(s2t_rewards) + 1), s2t_rewards, linestyle='-', label='S2T Rewards', markersize=4, alpha=0.8)
        plt.plot(range(1, len(extension_uniform_rewards_3) + 1), extension_uniform_rewards_3, linestyle='-', label='Uniform Rewards', markersize=4, alpha=0.8)
        #plt.plot(range(1, len(extension_gaussian_rewards_3) + 1), extension_gaussian_rewards_3, linestyle='-', label='Gaussian Rewards', markersize=4, alpha=0.8)
        #plt.plot(range(1, len(extension_gaussian_rewards_thigh) + 1), extension_gaussian_rewards_thigh, linestyle='-', label='Gaussian Rewards Thigh', markersize=4, alpha=0.8)

        # Add a mean reward line for each list
        #plt.axhline(y=np.mean(udr_rewards), color='r', linestyle='--', label='Mean UDR Rewards')
        plt.axhline(y=np.mean(t2t_rewards), color='g', linestyle='--', label='Mean T2T Rewards')
        #plt.axhline(y=np.mean(s2t_rewards), color='cyan', linestyle='--', label='Mean S2T Rewards')
        plt.axhline(y=np.mean(extension_uniform_rewards_3), color='b', linestyle='--', label='Mean Uniform Rewards')
        #plt.axhline(y=np.mean(extension_gaussian_rewards_3), color='m', linestyle='--', label='Mean Gaussian Rewards')
        #plt.axhline(y=np.mean(extension_gaussian_rewards_thigh), color='m', linestyle='--', label='Mean Gaussian Rewards Thigh')

        # Add titles and labels
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.xticks(range(1, max(len(udr_rewards), len(t2t_rewards)) + 1))

        # Adjust legend location
        plt.legend(loc='lower left', fontsize=10, frameon=True, shadow=True, borderpad=1)
        plt.grid(alpha=0.5, linestyle='--')

        # Show the plot
        plt.tight_layout()
        plt.savefig('./walker/figures/udr_3_delta')
        plt.show()
    