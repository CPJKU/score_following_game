from pynput import keyboard
from score_following_game.agents.optimal_agent import OptimalAgent

human_action = 1

def on_press(key):
    global human_action

    if key == keyboard.Key.left:
        human_action = 0

    if key == keyboard.Key.right:
        human_action = 2


class HumanAgent(OptimalAgent):

    def __init__(self, rl_pool):
        super(HumanAgent, self).__init__(rl_pool)
        keyboard.Listener(on_press=on_press).start()

    def select_action(self, state, train=False):
        global human_action

        action = human_action
        human_action = 1
        return action
