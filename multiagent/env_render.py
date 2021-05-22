import time

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


scenario_name = "simple_tag"
# load scenario from script
scenario = scenarios.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
env.render()
time.sleep(100)
