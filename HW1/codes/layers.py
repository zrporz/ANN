import numpy as np

def generate_array(p):
    # 生成一个大小为784的全零数组
    arr = np.zeros(784)
    
    # 计算需要设置为1的个数
    num_ones = int(p * 784)
    
    # 随机选择num_ones个位置，并将其设置为1
    arr[:num_ones] = 1
    
    # 打乱数组顺序
    np.random.shuffle(arr)
    
    return arr

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
	def __init__(self, name):
		super(Relu, self).__init__(name)

	def forward(self, input):
		self._saved_for_backward(input)
		return np.maximum(0, input)

	def backward(self, grad_output):
		input = self._saved_tensor
		return grad_output * (input > 0)

class Sigmoid(Layer):
	def __init__(self, name):
		super(Sigmoid, self).__init__(name)

	def forward(self, input):
		output = 1 / (1 + np.exp(-input))
		self._saved_for_backward(output)
		return output

	def backward(self, grad_output):
		output = self._saved_tensor
		return grad_output * output * (1 - output)

class Selu(Layer):
    lam = 1.0507
    alpha = 1.67326
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        mask = input > 0
        output = np.zeros(input.shape)
        output[mask] = self.lam * input[mask]
        output[mask==False] = self.lam * self.alpha * (np.exp(input[mask==False]) - 1)
        self._saved_for_backward(output)
        return output
        pass
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        output = self._saved_tensor
        mask = output > 0
        output[mask] = grad_output[mask] * self.lam
        output[mask==False] = grad_output[mask==False] * (output[mask==False]+ self.lam *self.alpha)
        return output
        pass
        # TODO END

class Swish(Layer):
    def __init__(self, name):
        super(Swish, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        output = input / (1+np.exp(-input)) # x * sigmoid(x)
        self._saved_for_backward(input)
        return output
        pass
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        input = self._saved_tensor
        output = input / (1+np.exp(-input)) # x * sigmoid(x)
        return grad_output * (output + (1-output) / (1+np.exp(-input))) # swish(x) + (1-swish(x)) * sigmoid(x)
        pass
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        output = 0.5 * input * (1 + np.tanh(np.sqrt(2/np.pi) * (input + 0.044715 * (input ** 3))))
        return output
        pass
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        delta = 1e-05
        input = self._saved_tensor
        def gelu(x):
             return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * (x ** 3))))
        return grad_output * (gelu(input+delta)- gelu(input))/delta
        pass
        # TODO END
    
    
class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std,ismask=False):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)
        self.ismask = ismask

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        if self.ismask:
            self.mask = generate_array(0.8)
            output = np.matmul(input,self.W*self.mask) + self.b * self.mask
        else:
            output = np.matmul(input,self.W) + self.b 
        return output
        pass
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        # print(grad_output.shape)
        input = self._saved_tensor
        if self.ismask:
            grad_output *= self.mask
        self.grad_W = np.matmul(input.T, grad_output)
        self.grad_b = np.sum(grad_output,axis=0)
        return np.matmul(grad_output, self.W.T)
        pass
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
