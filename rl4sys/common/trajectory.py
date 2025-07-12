"""
RL4Sys Base Classes

This module contains the core base classes for the RL4Sys framework.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import threading

from rl4sys.common.action import RL4SysAction

class RL4SysTrajectory():
    """
    A trajectory of actions taken by an RL4SysAgent.

    Attributes:
        actions (List[RL4SysAction]): list of actions taken in the trajectory.
        version (int): version of the model that generated this trajectory.
        invalid_mixed (bool): whether this trajectory contains actions from different versions.
    """
    def __init__(self, version: int):
        self.actions = []
        self.version = version
        self.invalid_mixed = False
        self.completed = False
        
    def add_action(self, action: RL4SysAction) -> None:
        """
        Add an action to the trajectory. If the action's version differs from the
        trajectory's version, mark the trajectory as invalid_mixed.
        """
        if action.version != self.version:
            self.invalid_mixed = True
        self.actions.append(action)

    def get_actions(self) -> List[RL4SysAction]:
        return self.actions

    def is_empty(self) -> bool:
        return len(self.actions) == 0

    def clear(self) -> None:
        self.actions = []
        self.invalid_mixed = False

    def is_valid(self) -> bool:
        return not self.invalid_mixed

    def mark_completed(self) -> None:
        """
        Mark the trajectory as completed.
        """
        self.completed = True

    def is_completed(self) -> bool:
        return self.completed
    
    def print_actions(self):
        for i, action in enumerate(self.actions):
            print(f"action {i}: {action.obs} {action.act} {action.rew}")