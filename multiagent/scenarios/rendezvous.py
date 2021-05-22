import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    """
    rendezvous
    """
    def make_world(self):
        world = World()
        # set any world properties
        world.dim_c = 2
        num_agents = 20
        world.num_agents = num_agents
        # add agents (evaders)
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.02
            agent.max_speed = 1.0
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
                agent.color = np.array([0.1, 0.2, 0.3])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    def reward(self, agent, world):
        dist = []
        for i in world.agents:
            for j in world.agents:
                delta_pos = i.state.p_pos - j.state.p_pos
                dis = np.sqrt(np.sum(np.square(delta_pos)))
                dist.append(dis)
        rew = -1.0 * np.max(dist)
        # for i in world.agents:
        #     if abs(i.state.p_pos[0]) > 1 or abs(i.state.p_pos[1]) > 1:
        #         rew -= 10.0
        return rew


    def observation(self, agent, world):
        """
        自己的位置和速度，和其他每个 agent 的相对位置和速度
        """
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos)
