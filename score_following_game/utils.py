
import os
import sys


def init_rl_imports():
    project_path = os.path.dirname(os.path.abspath(__file__))
    rl_path = os.path.join(project_path, "reinforcement_learning")
    sys.path.insert(1, rl_path)
