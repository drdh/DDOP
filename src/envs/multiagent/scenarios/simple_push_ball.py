import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, neighbor_obs=None):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_agents = 3
        num_landmarks = 1
        num_walls = 4

        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.size = 0.02
            agent.max_speed = 0.4
            agent.adversary = False

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = True
            landmark.size = 0.1
            landmark.initial_mass = 10

        # Add walls
        world.walls = [Wall() for _ in range(num_walls)]
        for i, wall in enumerate(world.walls):
            wall.name = 'wall %d' % i
            wall.movable = False
            wall.width = 0.05

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # Random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # Random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # Color for walls
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0.75, 0.25, 0])

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([0.75, 0.75])
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            while np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))) <= world.landmarks[0].size + 0.01:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.walls[0].state.start = np.array([-1, -1])
        world.walls[0].state.end = np.array([-1, 1])

        world.walls[1].state.start = np.array([-1, 1])
        world.walls[1].state.end = np.array([1, 1])

        world.walls[2].state.start = np.array([1, 1])
        world.walls[2].state.end = np.array([1, -1])

        world.walls[3].state.start = np.array([1, -1])
        world.walls[3].state.end = np.array([-1, -1])

        for xi in range(0, 4):
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

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = -np.sqrt(np.sum(np.square(world.landmarks[0].state.p_pos - np.array([-0.2, -0.2]))))
        return rew

    def observation(self, agent, world, env=None):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
