from __future__ import division
import numpy as np

class MSELoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        return np.mean(np.sum((target-input) ** 2,axis=1))
        pass
        # TODO END

    def backward(self, input, target):
		# TODO START
        '''Your codes here'''
        return (input - target) * 2 / input.shape[0]
        pass
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        exp_input = np.exp(input)
        return np.mean(np.sum(-target * np.log(exp_input / np.sum(exp_input,axis=1,keepdims=True)),axis=1),axis=0)
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        exp_input = np.exp(input)
        return -target + np.sum(target,axis=1,keepdims=True) / np.sum(exp_input,axis=1,keepdims=True) * exp_input
        pass
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        '''Your codes here'''
        x_tn = np.expand_dims(input[target==1],axis=1)
        return np.mean(np.sum(np.maximum(0, self.margin - x_tn + input),axis=1)) - self.margin
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        # print(input)
        mask_tn = (target == 1)
        result = np.zeros(input.shape)
        mask_no_zero = input - np.expand_dims(input[mask_tn],axis=1) + self.margin > 0
        result[mask_no_zero] = 1
        result[mask_tn] = -np.sum(mask_no_zero,axis=1) + 1
        return result
        pass
        # TODO END


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        exp_input = np.exp(input)
        # print(exp_input==0)
        softmax_input = exp_input / np.sum(exp_input,axis=1,keepdims=True) 
        one_minus_alpha = [1.0- self.alpha[i] for i in range(10)] 
        return -np.mean(np.sum((self.alpha * target + one_minus_alpha*(1-target))*((1-softmax_input)**self.gamma) * target * np.log(softmax_input),axis=1))
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        exp_input = np.exp(input)
        softmax_input = exp_input / np.sum(exp_input,axis=1,keepdims=True)
        one_minus_alpha = [1.0- self.alpha[i] for i in range(10)] 
        # output = (self.alpha * target + one_minus_alpha*(1-target))*target*(self.gamma *(1-softmax_input)**(self.gamma-1)*np.log(softmax_input) - (1-softmax_input)**self.gamma/softmax_input)
        output1 = -np.sum((self.alpha * target + one_minus_alpha*(1-target)) * target * (-softmax_input)*(-np.log(softmax_input) * self.gamma * (1-softmax_input)**(self.gamma-1)+(1-softmax_input)**self.gamma/softmax_input),axis=1,keepdims=True)*softmax_input
        
        output1 -= (self.alpha * target + one_minus_alpha*(1-target)) * target * (-np.log(softmax_input) * self.gamma * (1-softmax_input)**(self.gamma-1)+(1-softmax_input)**self.gamma/softmax_input) * softmax_input
        # output = output*(softmax_input - softmax_input * np.sum(softmax_input, axis=1, keepdims=True))
        # print(output==output1)
        return output1
        # return output1
        pass
        # TODO END
