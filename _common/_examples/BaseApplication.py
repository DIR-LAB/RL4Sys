from abc import ABC, abstractmethod


class ApplicationAbstract(ABC):
    def __init__(self, *args, **kwargs):
        super(ApplicationAbstract, self).__init__(*args, **kwargs)

    @abstractmethod
    def run_application(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_observation(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_performance_return(self, *args, **kwargs):
        pass
