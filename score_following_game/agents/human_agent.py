from pynput import keyboard

human_action = 1


def on_press(key):
    global human_action

    if key == keyboard.Key.left:
        human_action = 0

    if key == keyboard.Key.right:
        human_action = 2


class HumanAgent:

    def __init__(self):
        keyboard.Listener(on_press=on_press).start()

    def select_action(self, state, train=False):
        global human_action

        action = human_action
        human_action = 1
        return action
