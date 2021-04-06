import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, neighbor_obs=None):
        world = World()
        # set any world properties first
        world.dim_p = 1
        num_agents = 2
        num_landmarks = 0
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(0, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        dists = np.sqrt(np.square((world.agents[0].state.p_pos[0] - 0.5)) + np.square((world.agents[1].state.p_pos[0] - 0.5)))
        rew  = -dists

        return (rew)

    def reward(self, agent, world):
        dists = np.sqrt(np.square((world.agents[0].state.p_pos[0] - 0.5)) + np.square((world.agents[1].state.p_pos[0] - 0.5)))
        rew  = -dists

        return rew

    def observation(self, agent, world):
        other_pos = 0
        for a in world.agents:
            if a is not agent:
                other_pos = a.state.p_pos[0]
                break

        return np.array([agent.state.p_pos[0], other_pos])
