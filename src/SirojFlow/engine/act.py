import numpy as np
from ._sirojflow import SirojFlow

class Activation(SirojFlow):
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, next_grad):
        raise NotImplementedError
    
    
class ReLU(Activation):
    def forward(self, x):
        val = np.maximum(0, x)
        self.input = x
        self.out = val
        return val
    
    def backward(self, next_grad):
        grad = np.where(self.input > 0.0, 1.0, 0.0)
        self.grad_next = next_grad * grad
        return self.grad_next
    
    def __repr__(self):
        return "ReLU()"
    

class Sigmoid(Activation):  #mul
    def forward(self, x):
        # print(type(x))
        val = 1 / (1 + np.exp(-1*x))
        self.input = x
        self.out = val
        return val
    
    def backward(self, next_grad):
        grad = self.out*(1-self.out)
        self.grad_next = next_grad * grad

        return self.grad_next
    
    def __repr__(self):
        return "Sigmoid()"
    

class Softmax(Activation): #matmul
    def forward(self, x):
        final_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(final_x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.input = x
        self.out = probs
        return probs
        
    def backward(self, next_grad):
        s_times_g = self.out * next_grad
        
        dot = np.sum(s_times_g, axis=1, keepdims=True)      
        self.grad_next = s_times_g - self.out * dot         
        return self.grad_next

    def __repr__(self):
        return "Softmax()" 
    

class Tanh(Activation):
    def forward(self, x):
        val = np.tanh(x)
        self.input = x
        self.out = val
        return val
    
    def backward(self, next_grad):
        grad = 1 - self.out**2
        self.grad_next = next_grad * grad
        return self.grad_next
    
    def __repr__(self):
        return "Tanh()"


class HeavySide(Activation):
    def forward(self, x):
        val = np.where(x > 0.0, 1.0, 0.0)
        self.input = x
        self.out = val
        return val
    
    def backward(self, next_grad):
        self.grad_next = np.zeros_like(self.input)
        return self.grad_next

    def __repr__(self):
        return "HeavySide()"


class Swish(Activation):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        val = x * (1 / (1 + np.exp(-x*self.beta)))
        self.input = x
        self.out = val
        return val
    
    def backward(self, next_grad):
        sig_val = 1 / (1 + np.exp(-self.beta*self.input))
        sig_grad = sig_val*(1-sig_val)

        grad = sig_val + self.beta * self.input * sig_grad
        self.grad_next = next_grad * grad
        return self.grad_next

    def __repr__(self):
        return f"Swish(beta={self.beta})"
    

class LeakyReLU(Activation):
    def __init__(self, alpha=1e-4):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        val = np.where(x > 0.0, x, self.alpha*x)
        self.input = x
        self.out = val
        return val
    
    def backward(self, next_grad):
        grad = np.where(self.input > 0.0, 1.0, self.alpha)
        self.grad_next = next_grad * grad
        return self.grad_next

    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"



