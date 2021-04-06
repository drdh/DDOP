import numpy as np
from envs.multiagent.core import World, Agent, Landmark
from envs.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, neighbor_obs=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 5
        num_landmarks = 1
        world.collaborative = True
        world.mean_reward = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark()]
        world.landmarks[0].name = 'landmark %d' % i
        world.landmarks[0].collide = False
        world.landmarks[0].movable = False
        world.landmarks[0].size = 0.15
        # make initial conditions
        self.dis = 0.2
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.25, 0.25, 0.25])
        world.landmarks[0].state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.array([
                np.random.uniform(max(-1, world.landmarks[0].state.p_pos[0] - self.dis), min(world.landmarks[0].state.p_pos[0] + self.dis, 1)),
                np.random.uniform(max(-1, world.landmarks[0].state.p_pos[1] - self.dis), min(world.landmarks[0].state.p_pos[1] + self.dis, 1)),
            ])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def set_world(self, world, initial_state):
        state_i = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = initial_state[state_i]
            state_i += 1
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.landmarks[0].state.p_pos = initial_state[state_i]
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        dists = np.array([np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[0].state.p_pos))) for a in world.agents])
        if (dists < 0.1).all():
            return 10
        else:
            return 0

    def observation(self, agent, world, env=None):
        # get positions of all entities in this agent's reference frame
        entity_pos = [world.landmarks[0].state.p_pos - agent.state.p_pos]

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

    def done(self, agent, world):
        dists = np.array([np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[0].state.p_pos))) for a in world.agents])
        if (dists < 0.1).all():
            return True
        else:
            return False
        # return False
