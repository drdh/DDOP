import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, neighbor_obs=None, neighbor_radius=None):
        world = World()

        self.gridworld = True
        self.gridworld_dim = 1
        self.gridworld_step = 0.01
        self.scale_factor = 1
        self.neighbor_obs = neighbor_obs
        self.neighbor_radius = neighbor_radius

        self.agent_identity = False
        self.landmark_identity = False
        self.wall_identity = False

        # set any world properties first
        world.dim_c = 2
        self.num_landmarks = 0
        self.num_agents = 2
        self.num_walls = 8

        world.collaborative = True
        # add agents
        world.agents = [Agent(max_speed=0.4) for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02 / self.scale_factor
            agent.adversary = False

        # Add landmarks
        # world.landmarks = [Landmark() for i in range(num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d' % i
        #     landmark.collide = False
        #     landmark.movable = False
        #     landmark.size = 0.03 / self.scale_factor

        # Add walls
        world.walls = [Wall() for _ in range(self.num_walls)]
        for i, wall in enumerate(world.walls):
            wall.name = 'wall %d' % i
            wall.movable = False
            wall.width = 0.07

            if i >= 4:
                wall.width = 0.02

        # Nearest observation paradigm parameters.
        if not self.neighbor_obs:
            self.num_near_l = 1
            self.num_near_w = 3
            self.num_near_a = 1

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        for i, wall in enumerate(world.walls):
            wall.color = np.array([0.75, 0.25, 0])

        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.agents[0].state.p_pos = np.array([0.88, 0.8]) / self.scale_factor
        world.agents[1].state.p_pos = np.array([-0.88, -0.8]) / self.scale_factor

        # Set positions for walls
        world.walls[0].state.start = np.array([1.13, 1.13]) / self.scale_factor
        world.walls[0].state.end = np.array([-1.13, 1.13]) / self.scale_factor

        world.walls[1].state.start = np.array([-1.13, 1.13]) / self.scale_factor
        world.walls[1].state.end = np.array([-1.13, -1.13]) / self.scale_factor

        world.walls[2].state.start = np.array([-1.13, -1.13]) / self.scale_factor
        world.walls[2].state.end = np.array([1.13, -1.13]) / self.scale_factor

        world.walls[3].state.start = np.array([1.13, -1.13]) / self.scale_factor
        world.walls[3].state.end = np.array([1.13, 1.13]) / self.scale_factor

        # Walls for tracks
        world.walls[4].state.start = np.array([-0.93, 0.86])
        world.walls[4].state.end = np.array([0.93, 0.86])

        world.walls[5].state.start = np.array([-0.93, 0.74])
        world.walls[5].state.end = np.array([0.93, 0.74])

        world.walls[6].state.start = np.array([-0.93, -0.86])
        world.walls[6].state.end = np.array([0.93, -0.86])

        world.walls[7].state.start = np.array([-0.93, -0.74])
        world.walls[7].state.end = np.array([0.93, -0.74])

        for xi in range(0, self.num_walls):
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
        rew = 0

        count = 0
        for agent in world.agents:
            if agent.state.p_pos[0] < -0.85:
                count += 1

        return 100 * (count == 2)

    def observation(self, agent, world, env=None):
        if self.neighbor_obs:
            near_landmark_pro = []
            near_wall_pro = []
            near_agent_pro = []

            # No identity, sorted by distance.
            # Nearby landmarks

            if self.landmark_identity:
                near_landmark_pro = [np.zeros(shape=[world.dim_p + 1 + 3]) for _ in range(self.num_landmarks)]
                landmark_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
                                 entity
                                 for entity in world.landmarks}

                sls = sorted(landmark_dict.items(), key=lambda x: x[0])

                for sl_index in range(self.num_landmarks):
                    if sls[sl_index][0] <= self.neighbor_radius:
                        near_landmark_pro[sl_index] = np.concatenate((sls[sl_index][1].state.p_pos - agent.state.p_pos,
                                                                      np.array([sls[sl_index][1].size]),
                                                                      sls[sl_index][1].color))
                    else:
                        break

            else:
                near_landmark_pro = [np.zeros(shape=[world.dim_p + 1]) for _ in range(self.num_landmarks)]
                landmark_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
                                 entity
                                 for entity in world.landmarks}

                sls = sorted(landmark_dict.items(), key=lambda x: x[0])

                for sl_index in range(self.num_landmarks):
                    if sls[sl_index][0] <= self.neighbor_radius:
                        near_landmark_pro[sl_index] = np.concatenate((sls[sl_index][1].state.p_pos - agent.state.p_pos,
                                                                      np.array([sls[sl_index][1].size])))
                    else:
                        break

            # Nearby walls ### Maybe distance should be between point and line.
            if self.wall_identity:
                near_wall_pro = [np.zeros(shape=[world.dim_p * 3 + 1 + 3]) for _ in range(self.num_walls)]
                wall_dict = {np.sqrt(np.sum(np.square(wall.state.p_pos - agent.state.p_pos))):
                             wall for wall in world.walls}

                sws = sorted(wall_dict.items(), key=lambda x: x[0])

                for sw_index in range(self.num_walls):
                    if sws[sw_index][0] < self.neighbor_radius:
                        near_wall_pro[sw_index] = \
                            np.concatenate((sws[sw_index][1].state.start,
                                            sws[sw_index][1].state.end,
                                            sws[sw_index][1].state.p_pos - agent.state.p_pos,
                                            np.array([sws[sw_index][1].width])))
                    else:
                        break

            else:
                near_wall_pro = [np.zeros(shape=[world.dim_p * 3 + 1]) for _ in range(self.num_walls)]
                wall_dict = {np.sqrt(np.sum(np.square(wall.state.p_pos - agent.state.p_pos))):
                             wall for wall in world.walls}

                sws = sorted(wall_dict.items(), key=lambda x: x[0])

                for sw_index in range(self.num_walls):
                    if sws[sw_index][0] < self.neighbor_radius:
                        near_wall_pro[sw_index] =\
                            np.concatenate((sws[sw_index][1].state.start,
                                            sws[sw_index][1].state.end,
                                            sws[sw_index][1].state.p_pos - agent.state.p_pos,
                                            np.array([sws[sw_index][1].width]),
                                            sws[sw_index][1].color))
                    else:
                        break

            # Nearby agents

            if self.agent_identity:
                near_agent_pro = [np.zeros(shape=[world.dim_p + 1 + 3]) for _ in range(self.num_agents)]

                other_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
                              entity for entity in world.agents}

                sas = sorted(other_dict.items(), key=lambda x: x[0])

                for sa_index in range(self.num_agents):
                    if sas[sa_index][0] < self.neighbor_radius:
                        near_agent_pro[sa_index] = np.concatenate((sas[sa_index][1].state.p_pos - agent.state.p_pos,
                                                                   np.array([sas[sa_index][1].size]),
                                                                   sas[sa_index][1].color))
                    else:
                        break
            else:
                near_agent_pro = [np.zeros(shape=[world.dim_p + 1]) for _ in range(self.num_agents)]

                other_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
                              entity
                              for entity in world.agents}

                sas = sorted(other_dict.items(), key=lambda x: x[0])

                for sa_index in range(self.num_agents):
                    if sas[sa_index][0] < self.neighbor_radius:
                        near_agent_pro[sa_index] = np.concatenate((sas[sa_index][1].state.p_pos - agent.state.p_pos,
                                                                   np.array([sas[sa_index][1].size])))
                    else:
                        break

            # Result
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.array([agent.size])] +
                                  near_landmark_pro + near_wall_pro + near_agent_pro)

        else:
            # get positions of all entities in this agent's reference frame
            landmark_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
                                 entity.state.p_pos - agent.state.p_pos
                             for entity in world.landmarks}

            sls = sorted(landmark_dict.items(), key=lambda x: x[0])
            near_landmark_pos = [sls[sl_index][1] for sl_index in range(self.num_near_l)]

            # Nearby walls
            wall_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
                         wall_i
                         for wall_i, entity in enumerate(world.walls)}

            sws = sorted(wall_dict.items(), key=lambda x: x[0])
            near_wall_pos = [np.concatenate((world.walls[sws[sw_index][1]].state.start,
                                             world.walls[sws[sw_index][1]].state.end,
                                             world.walls[sws[sw_index][1]].state.p_pos - agent.state.p_pos))
                             for sw_index in range(self.num_near_w)]

            # Nearby agents
            other_dict = {np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))):
                              entity.state.p_pos - agent.state.p_pos
                          for entity in world.agents}

            sas = sorted(other_dict.items(), key=lambda x: x[0])
            near_agent_pos = [sas[sa_index][1] for sa_index in range(self.num_near_a)]

            # If identity is needed
            # entity colors
            # entity_color = []
            # for entity in world.landmarks:  # world.entities:
            #     entity_color.append(entity.color)
            # communication of all other agents

            # comm = []
            # for other in world.agents:
            #     if other is agent:
            #         continue
            #
            #     comm.append(other.state.c)

            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + np.array([agent.size]) +
                                  near_landmark_pos + near_wall_pos + near_agent_pos)

    def done(self, agent, world):
        count = 0
        for agent in world.agents:
            if agent.state.p_pos[0] < -0.85:
                count += 1

        return count == 2