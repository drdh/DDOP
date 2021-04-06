from envs.multiagentenv import MultiAgentEnv
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from gym import spaces

class Robosuite(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)


        controller_configs = load_controller_config(default_controller=self.args.default_controller)
        self.env = suite.make(
            self.args.scenario,
            robots=self.args.robots,
            gripper_types=self.args.gripper_types,
            controller_configs=controller_configs,
            env_configuration=self.args.env_configuration,
            has_renderer=self.args.has_renderer,
            has_offscreen_renderer=self.args.has_offscreen_renderer,
            control_freq=self.args.control_freq,
            horizon=self.args.horizon,
            use_object_obs=self.args.use_object_obs,
            use_camera_obs=self.args.use_camera_obs,
            reward_shaping=self.args.reward_shaping,
        )

        self.obs_list = self.env.reset()
        self.n_agents = self.args.num_joints # TODO: consider gripper,multi-robot
        if self.args.has_gripper:
            self.n_agents += 1
        self.n_actions = self.args.action_dims
        self.episode_limit = self.args.horizon
        self.action_space = [spaces.Box(low=np.array([self.env.action_spec[0][0]]), # TODO: check whether all the action belong to [-1,1]
                                        high=np.array([self.env.action_spec[1][0]]), dtype=np.float32)
                             for _ in range(self.n_agents)]
        self.observation_space = [spaces.Box(low=np.ones(10)*(-1), high=np.ones(10), dtype=np.float32)
                                  for _ in range(self.n_agents)]

        self.steps = 0
        self.reward_this_episode = 0

    def step(self, actions):
        self.obs_list, reward, done, info = self.env.step(actions.reshape(-1))
        self.steps += 1
        self.reward_this_episode += reward

        if done:
            info["episode_reward"] = self.reward_this_episode
            if self.steps < self.episode_limit:
                info["episode_limit"] = False   # the next state will be masked out
            else:
                info["episode_limit"] = True    # the next state will not be masked out
        return reward/2, done, info
    # 10, (50, 100), 5, 2


    def get_obs(self):
        return [self.get_obs_agent(a) for a in range(self.n_agents)]

    def get_obs_agent(self, agent_id):# TODO: consider gripper,multi-object,multi-robot
        if self.args.has_gripper and agent_id == self.n_agents - 1:
            obs = np.concatenate([self.obs_list['robot0_eef_pos'],
                                  self.obs_list['robot0_eef_quat'],
                                  self.obs_list['object-state'],
                                  np.zeros(3),
                                  self.obs_list['robot0_gripper_qpos'],
                                  self.obs_list['robot0_gripper_qvel'],
                                  np.ones(1),
                                  ])
        else:
            obs = np.concatenate([self.obs_list['robot0_eef_pos'],
                                  self.obs_list['robot0_eef_quat'],
                                  self.obs_list['object-state'],
                                  self.obs_list['robot0_joint_pos_cos'][agent_id:agent_id + 1],
                                  self.obs_list['robot0_joint_pos_sin'][agent_id:agent_id + 1],
                                  self.obs_list['robot0_joint_vel'][agent_id:agent_id + 1],
                                  np.zeros(5),
                                  ])
        return obs

    def get_obs_size(self):
        return len(self.get_obs_agent(0))

    def get_state(self): # TODO: consider gripper,multi-object,multi-robot
        if self.args.has_gripper:
            state = np.concatenate([self.obs_list['robot0_joint_pos_cos'],
                                    self.obs_list['robot0_joint_pos_sin'],
                                    self.obs_list['robot0_joint_vel'],
                                    self.obs_list['robot0_eef_pos'],
                                    self.obs_list['robot0_eef_quat'],
                                    self.obs_list['object-state'],
                                    self.obs_list['robot0_gripper_qpos'],
                                    self.obs_list['robot0_gripper_qvel'],
                                    ])
        else:
            state = np.concatenate([self.obs_list['robot0_joint_pos_cos'],
                                    self.obs_list['robot0_joint_pos_sin'],
                                    self.obs_list['robot0_joint_vel'],
                                    self.obs_list['robot0_eef_pos'],
                                    self.obs_list['robot0_eef_quat'],
                                    self.obs_list['object-state']
                                    ])
        return state

        # np.concatenate(self.get_obs())

    def get_state_size(self):
        return len(self.get_state())

    def get_avail_actions(self):
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def get_avail_agent_actions(self, agent_id):
        return np.ones(shape=(self.n_actions,))

    def get_total_actions(self): # dims
        return self.n_actions

    def get_stats(self):
        return {}

    def get_agg_stats(self, stats):
        return {}

    def reset(self):
        self.obs_list = self.env.reset()
        self.steps = 0
        self.reward_this_episode = 0

    def render(self):
        self.env.render()

    def close(self):
        raise NotImplementedError

    def seed(self, seed):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info