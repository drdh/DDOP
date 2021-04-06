import copy
import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, neighbor_obs=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        num_walls = 8
        world.collaborative = True

        self.door_open = False

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03

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
        self.door_open = False
        # Color for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # Color for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # Color for walls
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0.75, 0.25, 0])

        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.array([0., 0.])
            agent.state.p_pos[0] = np.random.uniform(-1, -0.1)
            agent.state.p_pos[1] = np.random.uniform(-1, 1)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Place landmarks
        world.landmarks[0].state.p_pos = np.array([-0.8, -0.8])
        world.landmarks[1].state.p_pos = np.array([0.8, 0.8])

        # Set fixed states for walls
        world.walls[0].state.start = np.array([0.0, -1.0])
        world.walls[0].state.end = np.array([0.0, 1.0])

        world.walls[1].state.start = np.array([0.0, 1.0])
        world.walls[1].state.end = np.array([0.0, -1.0])

        world.walls[2].state.start = np.array([world.walls[1].width + 0.001, 1])
        world.walls[2].state.end = np.array([1 - 0.001 - world.walls[1].width, 1])

        world.walls[3].state.start = np.array([-world.walls[1].width - 0.001, 1])
        world.walls[3].state.end = np.array([-1 + 0.001 + world.walls[1].width, 1])

        world.walls[4].state.start = np.array([world.walls[1].width + 0.001, -1])
        world.walls[4].state.end = np.array([1 - 0.001 - world.walls[1].width, -1])

        world.walls[5].state.start = np.array([-world.walls[1].width - 0.001, -1])
        world.walls[5].state.end = np.array([-1 + 0.001 + world.walls[1].width, -1])

        world.walls[6].state.start = np.array([1, 1])
        world.walls[6].state.end = np.array([1, -1])

        world.walls[7].state.start = np.array([-1, 1])
        world.walls[7].state.end = np.array([-1, -1])

        for xi in range(0, 8):
            world.walls[xi].state.p_pos = np.mean([world.walls[xi].state.start, world.walls[xi].state.end], axis=0)

    def benchmark_data(self, agent, world):
        rew = 0

        dist1 = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        dist2 = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[1].state.p_pos)))

        if dist1 < 0.1:
            rew += 0.1

        if dist2 < 0.1:
            rew += 10

        return rew

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0

        dist1 = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        dist2 = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[1].state.p_pos)))

        if dist1 < world.landmarks[0].size + world.agents[0].size:
            rew += 0.1

        if dist2 < world.landmarks[0].size + world.agents[0].size:
            rew += 10

        return rew

    def observation(self, agent, world, env=None):
        # Change state
        dist01 = np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - world.landmarks[0].state.p_pos)))
        dist02 = np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - world.landmarks[1].state.p_pos)))

        dist11 = np.sqrt(np.sum(np.square(world.agents[1].state.p_pos - world.landmarks[0].state.p_pos)))
        dist12 = np.sqrt(np.sum(np.square(world.agents[1].state.p_pos - world.landmarks[1].state.p_pos)))

        dist = np.array([dist01, dist02, dist11, dist12])

        if (dist < 0.1).any():
            if not self.door_open:
                # print('Open Door')
                self.open_door(world, env)
                self.door_open = True
        else:
            if self.door_open:
                # print('Close Door')
                self.close_door(world, env)
                self.door_open = False

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        for entity in world.walls:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        # entity colors
        entity_color = []
        for entity in world.walls:  # world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue

            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def open_door(self, world, env):
        env._reset_render()
        world.walls[0].state.end[1] = -0.1
        world.walls[0].state.p_pos[1] = -0.55

        world.walls[1].state.end[1] = 0.1
        world.walls[1].state.p_pos[1] = 0.55

    def close_door(self, world, env):
        env._reset_render()
        world.walls[0].state.end[1] = 1.0
        world.walls[0].state.p_pos[1] = 0.0

        world.walls[1].state.end[1] = -1.0
        world.walls[1].state.p_pos[1] = 0.0

    def calc_novelty(self, env, world, int_rew_func):
        # Save state of movable entities
        agents = [copy.deepcopy(agent) for agent in world.agents]

        # Calculate novelty
        world.agents[0].state.p_pos = np.array([-0.8, -0.8])
        world.agents[0].state.p_vel = np.array([0, 0])
        world.agents[1].state.p_vel = np.array([0, 0])

        x_c = [0.1 * t for t in range(-9, 10)]
        y_c = [0.1 * t for t in range(-9, 10)]

        obs0s = []
        obs1s = []
        action_n_i = []
        for x in x_c:
            for y in y_c:
                world.agents[1].state.p_pos = np.array([x, y])
                obs0 = self.observation(world.agents[0], world, env)
                obs1 = self.observation(world.agents[1], world, env)

                obs0s.append(obs0)
                obs1s.append(obs1)
                action_n_i.append(np.zeros(shape=[5]))

        obs_n = [np.array(obs0s), np.array(obs1s)]
        int_rew = int_rew_func(*(obs_n + [np.array(action_n_i)]))

        # Restore state of movable entities
        for i in range(len(agents)):
            world.agents[i] = copy.deepcopy(agents[i])

        # Restore door state
        dist01 = np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - world.landmarks[0].state.p_pos)))
        dist02 = np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - world.landmarks[1].state.p_pos)))

        dist11 = np.sqrt(np.sum(np.square(world.agents[1].state.p_pos - world.landmarks[0].state.p_pos)))
        dist12 = np.sqrt(np.sum(np.square(world.agents[1].state.p_pos - world.landmarks[1].state.p_pos)))

        dist = np.array([dist01, dist02, dist11, dist12])

        if (dist < 0.1).any():
            if not self.door_open:
                # print('Open Door')
                self.open_door(world, env)
                self.door_open = True
        else:
            if self.door_open:
                # print('Close Door')
                self.close_door(world, env)
                self.door_open = False

        return int_rew

    def done(self, agent, world):
        standard = [(np.sqrt(np.sum(np.square(a_agent.state.p_pos - world.landmarks[1].state.p_pos))) < 0.1)
                    for a_agent in world.agents]
        return np.array(standard).all()
