from abc import ABC, abstractmethod


class ApplicationAbstract(ABC):
    """
    Abstract class for designing and implementing new applications that can interface with RL4Sys.
    """
    def __init__(self, *args, **kwargs):
        super(ApplicationAbstract, self).__init__(*args, **kwargs)

    @abstractmethod
    def run_application(self, *args, **kwargs):
        """
        Run the application's main loop where observations are collected, actions are taken,
        and rewards are calculated.
        """
        pass

    @abstractmethod
    def build_observation(self, *args, **kwargs):
        """
        Build the observation for the current step to be passed through the RL4Sys Agent in run_application.
        """
        pass

    @abstractmethod
    def calculate_performance_return(self, *args, **kwargs):
        """
        Calculate the performance return for RL4Sys' flag_last_action reward value.
        """
        pass
