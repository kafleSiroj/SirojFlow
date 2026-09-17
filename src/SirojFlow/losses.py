import numpy as np

class Loss:
    def __init__(self, last_lay):
        self.epsilon = 1e-8
        self._label = None
        self._pred = None
        self._n = None
        self.last_lay = last_lay
        self._backward = None
    
    def __call__(self, pred, label):
        """
            
            ```
            pred: prediction/ output of model
            label: true output
            ```
            ---

            ### Usage
        
            ```
            for x_batch, y_batch in data_loader:
                ...
                loss = criterion(pred, y_batch)
                criterion.backward()
                ...
                
            ```

        """
        self._n = len(label)
        self._label = label
        return self.forward(pred, label)
    
    def forward(self, pred, label):
        raise NotImplementedError

    def backward(self):
        """
        `critetion.backward()` calls the `.backward()` of the loss function, this calculates gradient of loss with respect to output of model `prediction`, this gets cached as `grad_next` of the last layer of the model, which is passed into the loss object during criterion defination `criterion = <Loss>(model.layers[-1])`
        """
        raise NotImplementedError
    


class MSELoss(Loss):
    """
        ```
            last_lay: last layer of the model
        ```
        ### Usage
        ```
            criterion = MSELoss(model.layers[-1])
        ```
    """

    def forward(self, pred, label):
        loss = np.mean(np.square(pred - label))
        self._pred = pred
        return loss
    
    def backward(self):
        grad = 2 * (self._pred - self._label) / self._n
        self._backward = grad
        self.last_lay.grad_next = grad

    def __repr__(self):
        return "MSELoss()"
    

class MAELoss(Loss):
    """
        ```
            last_lay: last layer of the model
        ```
        ### Usage
        ```
            criterion = MAELoss(model.layers[-1])
        ```
    """

    def forward(self, pred, label):
        loss = np.mean(abs(label - pred))
        self._pred = pred
        return loss
    
    def backward(self):
        grad = np.sign(self._pred - self._label) / self._n
        self._backward = grad
        self.last_lay.grad_next = grad

    def __repr__(self):
        return "MAELoss()"


class BinaryCrossentropyLoss(Loss):
    """
        ```
            last_lay: last layer of the model
        ```
        ### Usage (for binary classification tasks)
        ```
            criterion = BinaryCrossentropyLoss(model.layers[-1])
        ```
    """
    def forward(self, pred, label):
        y_hat = np.clip(pred, self.epsilon, 1 - self.epsilon)
        loss = -np.mean(label*np.log(y_hat) + (1-label)*np.log(1-y_hat))
        self._pred = y_hat
        return loss
    
    def backward(self):
        grad = (self._pred-self._label) / self._n
        self._backward = grad
        self.last_lay.grad_next = grad

    def __repr__(self):
        return f"BinaryCrossentropyLoss(epsilon={self.epsilon})"


class SparseCategoricalCrossentropyLoss(Loss):
    """
        ```
            last_lay: last layer of the model
        ```
        ### Usage (for multiclass classification tasks)
        ```
            criterion = SparseCategoricalCrossentropyLoss(model.layers[-1])
        ```
    """
    def forward(self, pred, label):
        label = label.squeeze()
        self._label = label
        y_hat  = np.clip(pred, self.epsilon, 1 - self.epsilon)
        tclss = y_hat[np.arange(self._n), label]
        loss = -np.mean(np.log(tclss))
        self._pred = (y_hat, tclss)
        return loss

    def backward(self):
        grad = np.zeros_like(self._pred[0])
        grad[np.arange(self._n), self._label] = -1 / (self._pred[1] * self._n)
        self._backward = grad  #expects softmax
        self.last_lay.grad_next = grad

    def __repr__(self):
        return f"SparseCategoricalCrossentropyLoss(epsilon={self.epsilon})"
