import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, neighbor_obs=None):
        world = World()

        # Set any world properties first
        world.dim_c = 2

        num_agents = 2
        num_boundaries = 4
        num_walls = 4

        world.collaborative = True
        # add agents
        world.agents = [Agent(max_speed=0.025) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02

        # Add walls
        world.walls = [Wall() for _ in range(num_walls + num_boundaries)]
        for i in range(num_boundaries):
            world.walls[i].name = 'boundary %d' % i
            world.walls[i].movable = False
            world.walls[i].width = 0.15

        for i in range(num_boundaries, num_walls + num_boundaries):
            world.walls[i].name = 'wall %d' % i
            world.walls[i].movable = False
            world.walls[i].width = 0.01

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks

        # Color for walls
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0.75, 0.25, 0])

        # set random initial states
        # set random initial states

        world.agents[0].state.p_pos = np.array([0.88, 0.8])
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        world.agents[0].state.c = np.zeros(world.dim_c)

        world.agents[1].state.p_pos = np.array([-0.88, -0.8])
        world.agents[1].state.p_vel = np.zeros(world.dim_p)
        world.agents[1].state.c = np.zeros(world.dim_c)

        # Set fixed states for walls
        world.walls[0].state.start = np.array([1.13, 1.13])
        world.walls[0].state.end = np.array([-1.13, 1.13])

        world.walls[1].state.start = np.array([-1.13, 1.13])
        world.walls[1].state.end = np.array([-1.13, -1.13])

        world.walls[2].state.start = np.array([-1.13, -1.13])
        world.walls[2].state.end = np.array([1.13, -1.13])

        world.walls[3].state.start = np.array([1.13, -1.13])
        world.walls[3].state.end = np.array([1.13, 1.13])

        #
        world.walls[4].state.start = np.array([-0.93, 0.86])
        world.walls[4].state.end = np.array([0.93, 0.86])

        world.walls[5].state.start = np.array([-0.93, 0.74])
        world.walls[5].state.end = np.array([0.93, 0.74])

        world.walls[6].state.start = np.array([-0.93, -0.86])
        world.walls[6].state.end = np.array([0.93, -0.86])

        world.walls[7].state.start = np.array([-0.93, -0.74])
        world.walls[7].state.end = np.array([0.93, -0.74])

        for xi in range(0, 8):
            world.walls[xi].state.p_pos = np.mean([world.walls[xi].state.start, world.walls[xi].state.end], axis=0)

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

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world, env=None):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        count = 0

        for agent in world.agents:
            if agent.state.p_pos[0] < -0.9:
                count += 1

        rew = 0

        if count == 2:
            rew += 1

        return rew

    def observation(self, agent, world, env=None):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.walls:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.walls:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
