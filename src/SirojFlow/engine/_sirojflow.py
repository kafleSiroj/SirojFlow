import numpy as np

class SirojFlow:
    """
    Base Class for the framework
    """
    def __init__(self):
        self.params = {}
        self.layers = []
        self.out = None
        self.input = None
        self.grad_next = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray):
        return self.forward(x)
    
    def _register_params(self, weight, bias): 
        if "weights" not in self.params:
            self.params["weights"] = []
            self.params["biases"] = []
        
        self.params["weights"].append(weight)
        self.params["biases"].append(bias)

    def backward(self, next_grad: np.ndarray):
        raise NotImplementedError
