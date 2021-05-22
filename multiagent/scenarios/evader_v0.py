import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    """
    chaser and evader
    """
    def make_world(self):
        world = World()
        # set any world properties
        world.dim_c = 2
        num_agents = 5
        num_evaders = 1
        world.num_agents = num_agents
        # add agents (evaders)
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.evader = True if i < num_evaders else False  # 逃跑者
            agent.size = 0.02
            agent.max_speed = 1.0

            
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if agent.evader is True:
                agent.color = np.array([0.7, 0.1, 0.1])
            else:
                agent.color = np.array([0.1, 0.1, 0.7])
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

    def good_agents(self, world):
        return [agent for agent in world.agents if agent.evader is not True]

    def evaders(self, world):
        return [agent for agent in world.agents if agent.evader is True]

    def reward(self, agent, world):
        return self.evader_reward(agent, world) if agent.evader else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        """
        agent 奖励分两部分，共享的和私有的
        共享奖励为逃跑者的速度×-1
        私有奖励为是否进入逃跑者范围
        碰撞奖励 -1
        """
        good_agents = self.good_agents(world)
        evaders = self.evaders(world)
        rew = 0
        share_reward = 0
        private_rew = 0
        view = 0.12
        e_vel = []

        if agent.collide:
            for a in world.agents:
                if a == agent: continue
                if self.is_collision(a, agent):
                    rew -= 1

        for e in evaders:
            e_vel.append(e.state.p_vel)
            delta_pos = agent.state.p_pos - e.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            if dist < view:
                private_rew += 1.0
        for i in e_vel:
            absv = np.sqrt(np.sum(np.square(i)))
            share_reward -= absv * 1.0
        rew = share_reward + private_rew
        return rew

    def evader_reward(self, agent, world):
        """
        奖励为距离最近的 agent 的距离
        出界 -100
        被所有 agent 看到时，奖励为速度 × -1
        """
        good_agents = self.good_agents(world)
        evaders = self.evaders(world)
        rew = 0
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in good_agents]
        rew += min(dists)
        if abs(agent.state.p_pos[0]) > 1 or abs(agent.state.p_pos[1]) > 1:
            rew -= 100.0
        view = 0.12
        safe_num = 3
        view_num = 0
        for a in good_agents:
            delta_pos = a.state.p_pos - agent.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            if dist < view:
                view_num += 1
        if  view_num == len(good_agents):
            absv = np.sqrt(np.sum(np.square(agent.state.p_vel)))
            rew -= 1.0 * absv
        return rew

    def observation(self, agent, world):
        good_agents = self.good_agents(world)
        evaders = self.evaders(world)

        evader_pos = []
        for evader in evaders:
            evader_pos.append(evader.state.p_pos - agent.state.p_pos)

        comm = []
        other_pos = []
        for other in good_agents:
            if other is agent:continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + evader_pos + other_pos)
