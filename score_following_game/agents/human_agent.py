
from __future__ import print_function

from score_following_game.agents.optimal_agent import Agent
from pynput import keyboard

human_action = 1


def on_press(key):
    global human_action
    # print("Key press event")

    try:
        if key.char == 'w':
            human_action = 2

        if key.char == 's':
            human_action = 0

    except AttributeError:
        print('special key {0} pressed'.format(key))


class HumanAgent(Agent):

    def __init__(self):
        super(HumanAgent, self).__init__()
        self.cumulated_reward = 0

        keyboard.Listener(on_press=on_press).start()

    def perform_action(self, state):
        super(HumanAgent, self).perform_action(state)

        global human_action

        action = human_action
        human_action = 1
        return action


# from asyncore import file_dispatcher, loop
#
# class InputDeviceDispatcher(file_dispatcher):
#     from evdev import InputDevice, categorize, ecodes, KeyEvent
#
#     def __init__(self, device):
#         self.device = device
#         file_dispatcher.__init__(self, device)
#
#     def recv(self, ign=None):
#         return self.device.read()
#
#     def handle_read(self):
#
#         global human_action
#
#         for event in self.recv():
#             if event.type == ecodes.EV_KEY:
#                 event = categorize(event)
#                 if event.keystate == KeyEvent.key_down:
#                     print(event, event.keycode)
#                     # down BTN_THUMB, down BTN_THUMB2
#
#                     if event.keycode == "BTN_THUMB":
#                         human_action = 1
#                     elif event.keycode == "BTN_THUMB2":
#                         human_action = 2
#
# class HumanGamePadAgent(Agent):
#     from evdev import InputDevice, categorize, ecodes, KeyEvent
#
#     def __init__(self):
#         super(HumanGamePadAgent, self).__init__()
#         self.cumulated_reward = 0
#
#         dev = InputDevice('/dev/input/event16')
#         InputDeviceDispatcher(dev)
#         loop()
#
#     def perform_action(self, state):
#         super(HumanGamePadAgent, self).perform_action(state)
#
#         global speed, human_action
#
#         reward, timestep, current_speed = state
#
#         self.cumulated_reward += reward
#
#         action = human_action
#         human_action = 0
#
#         return action
#
#
# if __name__ == '__main__':
#     """ main """
#     import time
#     human_agent = HumanAgent()
#     while True:
#         pass
#
#     # HumanGamePadAgent()
#     # loop()
