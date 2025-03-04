"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0


    def set_random_parameters(self, thigh_mass, leg_mass, foot_mass, delta_thigh_mass, delta_leg_mass, delta_foot_mass, distribution_type='uniform'):
        """Set random masses"""

        self.set_parameters(self.sample_parameters(thigh_mass, leg_mass, foot_mass, delta_thigh_mass, delta_leg_mass, delta_foot_mass, distribution_type))


    def sample_parameters(self, thigh_mass, leg_mass, foot_mass, delta_thigh_mass, delta_leg_mass, delta_foot_mass, distribution_type='uniform'):
        """Sample masses according to a domain randomization distribution"""

        if distribution_type == 'uniform':     

            thigh_mass_sample = np.random.uniform(thigh_mass-float(thigh_mass * delta_thigh_mass), thigh_mass+float(thigh_mass * delta_thigh_mass)) 
            leg_mass_sample = np.random.uniform(leg_mass-float(leg_mass * delta_leg_mass), leg_mass+float(leg_mass * delta_leg_mass))
            foot_mass_sample = np.random.uniform(foot_mass-float(foot_mass * delta_foot_mass), foot_mass+float(foot_mass * delta_foot_mass))

        elif distribution_type == 'gaussian':

            # I don't want negative masses
            thigh_mass_sample = np.maximum(0, np.random.normal(thigh_mass, thigh_mass * delta_thigh_mass))
            leg_mass_sample = np.maximum(0, np.random.normal(leg_mass, leg_mass * delta_leg_mass))
            foot_mass_sample = np.maximum(0, np.random.normal(foot_mass, foot_mass * delta_foot_mass))

        else:
            print('-'*10, 'Unknown distribution')

        return [thigh_mass_sample, leg_mass_sample, foot_mass_sample]


    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, task, verbose=False):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[2] = task[0]
        self.sim.model.body_mass[3] = task[1]
        self.sim.model.body_mass[4] = task[2]
        if verbose:
            print("Masses setted to : %.2f, %.2f, %.2f, %.2f"%(self.sim.model.body_mass[1], self.sim.model.body_mass[2], self.sim.model.body_mass[3], self.sim.model.body_mass[4]))


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

