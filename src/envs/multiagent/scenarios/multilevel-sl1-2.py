import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, neighbor_obs=None):
        self.scale_factor = 3
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        num_walls = 9

        world.collaborative = True
        # add agents
        world.agents = [Agent(max_speed=0.4) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02 / self.scale_factor
            agent.adversary = False

        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03 / self.scale_factor

        # Add walls
        world.walls = [Wall() for _ in range(num_walls)]
        for i, wall in enumerate(world.walls):
            wall.name = 'wall %d' % i
            wall.movable = False
            wall.width = 0.07

            if i >= 4:
                wall.width = 0.05

        self.num_near_l = 2
        self.num_near_w = 3
        self.num_near_a = 1

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # Color for walls
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0.75, 0.25, 0])

        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.9, 0.9, world.dim_p)
            agent.state.p_pos[0] = np.random.uniform(-0.2, 0.2) / self.scale_factor
            agent.state.p_pos[1] = np.random.uniform(-0.9, -0.8) / self.scale_factor

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Set random initial states
        world.landmarks[0].state.p_pos = np.array([0.7, -0.85]) / self.scale_factor
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        world.landmarks[1].state.p_pos = np.array([0.85, -0.85]) / self.scale_factor
        world.landmarks[1].state.p_vel = np.zeros(world.dim_p)

        world.landmarks[2].state.p_pos = np.array([0.8, -0.75]) / self.scale_factor
        world.landmarks[2].state.p_vel = np.zeros(world.dim_p)

        # world.landmarks[2].state.p_pos = np.array([0.75, -0.85])
        # world.landmarks[2].state.p_vel = np.zeros(world.dim_p)

        # Set fixed states for walls
        world.walls[0].state.start = np.array([1.13, 1.13]) / self.scale_factor
        world.walls[0].state.end = np.array([-1.13, 1.13]) / self.scale_factor

        world.walls[1].state.start = np.array([-1.13, 1.13]) / self.scale_factor
        world.walls[1].state.end = np.array([-1.13, -1.13]) / self.scale_factor

        world.walls[2].state.start = np.array([-1.13, -1.13]) / self.scale_factor
        world.walls[2].state.end = np.array([1.13, -1.13]) / self.scale_factor

        world.walls[3].state.start = np.array([1.13, -1.13]) / self.scale_factor
        world.walls[3].state.end = np.array([1.13, 1.13]) / self.scale_factor

        # First seam
        world.walls[4].state.start = np.array([-0.5, -1.5]) / self.scale_factor
        world.walls[4].state.end = np.array([-0.5, -0.06]) / self.scale_factor

        world.walls[5].state.start = np.array([-0.5, 0.06]) / self.scale_factor
        world.walls[5].state.end = np.array([-0.5, 1.5]) / self.scale_factor

        # Second seam
        # world.walls[6].state.start = np.array([0.0, -0.99])
        # world.walls[6].state.end = np.array([0.0, -0.81])

        world.walls[6].state.start = np.array([0.0, -0.69]) / self.scale_factor
        world.walls[6].state.end = np.array([0.0, 1.5]) / self.scale_factor

        # Third seam
        world.walls[7].state.start = np.array([0.5, -1.5]) / self.scale_factor
        world.walls[7].state.end = np.array([0.5, -0.06]) / self.scale_factor

        world.walls[8].state.start = np.array([0.5, 0.06]) / self.scale_factor
        world.walls[8].state.end = np.array([0.5, 1.5]) / self.scale_factor

        for xi in range(0, 9):
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
        rew = 0

        dists = np.array([0 for _ in range(len(world.landmarks))])
        for i, l in enumerate(world.landmarks):
            dists[i] = (min([np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]) <
                        2 * agent.size)

        if dists.all():
            rew += 100

        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        return rew

    def observation(self, agent, world, env=None):
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

        sas = sorted(other_dict.items(), key=lambda x: x[0], reverse=True)
        near_agent_pos = [sas[sa_index][1] for sa_index in range(self.num_near_a)]

        # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # communication of all other agents

        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)

        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
        #                      near_landmark_pos + near_wall_pos + near_agent_pos + comm)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
                              near_landmark_pos + near_wall_pos)


    def done(self, agent, world):
        dists = np.array([0 for _ in range(len(world.landmarks))])
        for i, l in enumerate(world.landmarks):
            dists[i] = (min([np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]) <
                        2 * agent.size)

        return dists.all()
