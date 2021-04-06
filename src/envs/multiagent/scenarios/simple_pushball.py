import numpy as np
from envs.multiagent.core import World, Agent, Landmark
from envs.multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, neighbor_obs=None):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_agents = 5
        world.mean_reward = True
        num_landmarks = 1
        # num_walls = 4

        # world.collaborative = True
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02
            # agent.max_speed = 0.4
            agent.adversary = False
            agent.initial_mass = 2.

        # Add landmarks
        world.landmarks = [Landmark()]

        world.landmarks[0].name = 'landmark %d' % 0
        world.landmarks[0].collide = True
        world.landmarks[0].movable = True
        world.landmarks[0].size = 0.15
        world.landmarks[0].initial_mass = 16
        self.dis = 0.2

        # Add walls
        # world.walls = [Wall() for _ in range(num_walls)]
        # for i, wall in enumerate(world.walls):
        #     wall.name = 'wall %d' % i
        #     wall.movable = False
        #     wall.width = 0.05
        #
        # self.num_near_l = 1
        # self.num_near_w = 3
        # self.num_near_a = 2

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # Random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # Random properties for landmarks
        world.landmarks[0].color = np.array([0.25, 0.25, 0.25])
        # Color for walls
        # for i, wall in enumerate(world.walls):
        #     wall.color = np.array([0.75, 0.25, 0])

        world.landmarks[0].state.p_pos = np.array([np.random.uniform(-1, +1), np.random.uniform(0, +1)])
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.array([
                np.random.uniform(max(-1, world.landmarks[0].state.p_pos[0] - self.dis),
                                  min(world.landmarks[0].state.p_pos[0] + self.dis, 1)),
                np.random.uniform(world.landmarks[0].state.p_pos[1],
                                  min(world.landmarks[0].state.p_pos[1] + self.dis, 1)),
            ])

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # world.walls[0].state.start = np.array([-1, -1])
        # world.walls[0].state.end = np.array([-1, 1])
        #
        # world.walls[1].state.start = np.array([-1, 1])
        # world.walls[1].state.end = np.array([1, 1])
        #
        # world.walls[2].state.start = np.array([1, 1])
        # world.walls[2].state.end = np.array([1, -1])
        #
        # world.walls[3].state.start = np.array([1, -1])
        # world.walls[3].state.end = np.array([-1, -1])
        #
        # for xi in range(0, 4):
        #     world.walls[xi].state.p_pos = np.mean([world.walls[xi].state.start, world.walls[xi].state.end], axis=0)

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

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = -np.sqrt(np.sum(np.square(world.landmarks[0].state.p_pos - np.array([-0.2, -0.2]))))
    #     # if (agent.state.p_pos > 1.).any():
    #     #     rew -= 1
    #     return rew

    def reward(self, agent, world):
        if world.landmarks[0].state.p_pos[1] <= 0:
            return 10
        else:
            return 0

    def observation(self, agent, world, env=None):
        # get positions of all entities in this agent's reference frame
        # landmark_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
        #                      entity.state.p_pos - agent.state.p_pos
        #                  for entity in world.landmarks}
        #
        # sls = sorted(landmark_dict.items(), key=lambda x:x[0])
        # near_landmark_pos = [sls[sl_index][1] for sl_index in range(self.num_near_l)]

        # Nearby walls
        # wall_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
        #                  wall_i
        #              for wall_i, entity in enumerate(world.walls)}
        #
        # sws = sorted(wall_dict.items(), key=lambda x:x[0])
        # near_wall_pos = [np.concatenate((world.walls[sws[sw_index][1]].state.start,
        #                                  world.walls[sws[sw_index][1]].state.end,
        #                                  world.walls[sws[sw_index][1]].state.p_pos - agent.state.p_pos))
        #                  for sw_index in range(self.num_near_w)]

        # Nearby agents
        # other_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
        #                   entity.state.p_pos - agent.state.p_pos
        #               for entity in world.agents}
        #
        # sas = sorted(other_dict.items(), key=lambda x: x[0])
        # near_agent_pos = [sas[sa_index][1] for sa_index in range(self.num_near_a)]

        # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # communication of all other agents

        # comm = []
        # for other in world.agents:
        #     if other is agent: continue
        #     comm.append(other.state.c)

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_pos.append(world.landmarks[0].state.p_pos)
        entity_pos.append(world.landmarks[0].state.p_pos - agent.state.p_pos)
        entity_pos.append(world.landmarks[0].state.p_vel)

        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_pos.append(other.state.p_vel)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

    def done(self, agent, world):
        if world.landmarks[0].state.p_pos[1] <= 0:
            return True
        else:
            return False
