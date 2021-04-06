import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        num_agents = 10
        num_actions = 10
        world.collaborative = True
        world.num_actions = 10
        # add agents

        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i

        return world

    def reset_world(self, world):
        pass

    def reward(self, action_n):
        for action in action_n:
            a = np.argmax(action)

            if a != 0:
                return -1

        return 1

    def observation(self):
        return np.array([-1., 0., 1.])

    def done(self):
        return True
