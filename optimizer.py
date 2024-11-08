from abc import abstractmethod

class Optimizer:
    def __init__(self) -> None:
        self.FOM = None
        self.optimizer_parameter = None
        self.history = None
    
    @abstractmethod
    def solve(self):
        pass

    def _check_termination_criteria(self) -> bool:
        pass

    

class FOMOptimizer(Optimizer):
    def __init__(self) -> None:
        pass

    def solve(self):
        while not self._check_termination_criteria():
            pass


class ROMOptimizer(Optimizer):
    def __init__(self) -> None:
        self.FOM = None
        self.ROM = None
        self.reductor = None


    def solve(self):
        while not self._check_termination_criteria():
            pass

    def _initialize_optimization(self):    
        pass    

    def _get_regulatization(self):
        pass
    
    def _solve_subproblem(self):
        pass

    def _get_q_AGC(self):
        pass

    def _check_q_trial(self) -> bool:
        pass