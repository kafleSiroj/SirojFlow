import numpy as np
from ._sirojflow import SirojFlow

class Linear(SirojFlow):
    def __init__(self, in_features, out_features, init_type=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dW = None
        self.dB = None

        std = 0.01
        if init_type=="he":
            std = np.sqrt(2 / in_features)
        elif init_type=="xavier":
            std = np.sqrt(2 / (in_features + out_features))
        elif init_type=="zero":
            std = 0

        weight = np.random.randn(out_features, in_features) * std
        bias = np.zeros((out_features))
        self._register_params(weight, bias)

    def forward(self, x):
        val = x @ self.params["weights"][0].T + self.params["biases"][0]
        self.input = x 
        self.out = val
        return val
    
    def backward(self, next_grad):
        self.dW = next_grad.T @ self.input
        self.dB = np.sum(next_grad, axis=0)
        # print("dB:", self.dB)
        # print("call_back func: ", [self.dW, self.dB])

        self.grad_next = next_grad @ self.params["weights"][0]
        return self.grad_next
    
    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features})"


class Sequential(SirojFlow):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_params(self):
        all_params = {"weights": [], "biases": []}

        for layer in self.layers:
            if hasattr(layer, "params") and layer.params:
                all_params["weights"].extend(layer.params.get("weights", []))
                all_params["biases"].extend(layer.params.get("biases", []))

        return all_params
    
    def summary(self):
        lines = []

        lines.append("=" * 70)
        lines.append(f"{'Idx':<5}{'Layer':<45}{'Params':>15}")
        lines.append("=" * 70)

        total_params = 0

        for i, layer in enumerate(self.layers):

            params = 0

            if hasattr(layer, "weight"):
                params += layer.weight.size             

            if hasattr(layer, "bias"):
                params += layer.bias.size

            total_params += params

            lines.append(
                f"{i:<5}"
                f"{repr(layer):<45}"
                f"{params:>15,}"
            )

        lines.append("=" * 70)
        lines.append(f"{'Total Parameters':<50}{total_params:>20,}")
        lines.append("=" * 70)

        return "\n".join(lines)