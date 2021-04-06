from smac.env import MultiAgentEnv
import numpy as np


class MPEnv(MultiAgentEnv):
    def __init__(self,
                 scenario_name="8m",
                 benchmark=False,
                 episode_limit=50,
                 max_reward=10,
                 n_agents=1,
                 step_mul=None,
                 seed=None,
                 continuing_episode=False,
                 reward_sparse=False,
                 reward_only_positive=True,
                 reward_death_value=10,
                 reward_win=200,
                 reward_defeat=0,
                 reward_negative_scale=0.5,
                 reward_scale=False,
                 reward_scale_rate=20,
                 debug=False,
                 state_last_action=False,
                 shared_viewer=True,
                 matrix_game=False,
                 test_return=False):

        self.episode_limit = episode_limit
        self.continuing_episode = continuing_episode
        self.battles_game = 0
        self.timeouts = 0
        self._episode_count = 0
        self.n_agents = n_agents
        self._seed = seed

        # reward
        self.reward_scale = reward_scale
        self.max_reward = max_reward
        self.reward_scale_rate = reward_scale_rate

        from envs.multiagent.environment import MPEMultiAgentEnv
        from envs.multiagent.environment import MPEMultiAgentMatrixEnv
        import envs.multiagent.scenarios as scenarios

        self.debug = debug
        self.n = n_agents

        self._total_steps = 0
        self._episode_steps = 0
        self.matrix_game = matrix_game

        # load scenario from script
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        # create world
        world = scenario.make_world()

        # create multiagent environment
        if self.matrix_game:
            self.env = MPEMultiAgentMatrixEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                      done_callback=scenario.done)
        else:
            if benchmark:
                self.env = MPEMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                    scenario.benchmark_data)
            elif test_return:
                self.env = MPEMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                    set_callback=scenario.set_world)
            else:
                self.env = MPEMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                            done_callback=scenario.done)

        self.action_space = self.env.action_space
        self.action_dim = int(np.array([a_space.n for a_space in self.action_space]).max())
        self.obs_dim = int(np.array(self.env.obs_dims).max())

        self.last_action = np.zeros(self.action_dim)

        # Rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

        self.world = world

        self.scenario_name = scenario_name

    def step(self, actions):
        """ Returns reward, terminated, info """
        # actions = [int(a) for a in actions]
        if type(actions) == np.ndarray:
            self.last_action = actions
            corrected_actions = [action[:self.action_space[i].n] for i, action in enumerate(actions)]
        else:
            self.last_action = actions.detach().cpu().numpy()
            corrected_actions = [action[:self.action_space[i].n] for i, action in enumerate(actions.detach().cpu().numpy())]

        # Collect individual actions
        # sc_actions = []

        # Send action request

        reward, terminated, env_info = self.env.step(corrected_actions)

        self._total_steps += 1
        self._episode_steps += 1

        # Update units

        info = {"battle_won": False}
        # if terminated:
        #     info["battle_won"] = True
        #     self.battles

        if self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

            if self.scenario_name in ['simple_aggregation', 'simple_pushball']:
                if reward == 0:
                    reward -= 10

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        # if self.reward_scale:
        #     reward /= self.max_reward / self.reward_scale_rate

        return reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        _obs = self.env.get_all_obs()
        a = [np.concatenate([obs, np.zeros(self.obs_dim-obs.shape[0])]) for obs in _obs]
        return a

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_dim

    def get_state(self):
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
            np.float32
        )
        return obs_concat

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.obs_dim * self.n_agents

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1] * self.action_dim

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.action_dim

    def reset(self):
        self._episode_steps = 0
        self.env.reset()
        return self.get_obs(), self.get_state()

    def set(self, initial_state):
        self._episode_steps = 0
        self.env.set(initial_state)
        return self.get_obs(), self.get_state()

    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from .multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from .multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from .multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def close(self):
        return 0

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "timeouts": self.timeouts
        }
        return stats


if __name__ == '__main__':
    env = MPEnv(scenario_name='simple_spread')

    for i in range(100):
        reward, terminated, info = env.step([np.array([0.2, 0.1, 0, -0.1, -0.2]),
                                             np.array([0.2, 0.1, 0, -0.1, -0.2]),
                                             np.array([0.2, 0.1, 0, -0.1, -0.2])])
