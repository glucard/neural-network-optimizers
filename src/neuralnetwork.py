import numpy as np

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1 * (x > 0)

def softmax(x):
    expX = np.exp(x)
    return expX / np.sum(expX, axis=1, keepdims=True)

def batch(array, batch_size ):
    if batch_size == 0:
        return [array]

    
    array_size = len(array)
    batchs = [array[i:i+batch_size] for i in range(0, array_size, batch_size)]

    last_batch_len = len(batchs[-1])
    if last_batch_len != batch_size and batchs[-1] != batchs[0]:
        new_batchs_len = len(batchs) - 1
        for i, sample in enumerate(batchs[-1]):
            batchs[i % new_batchs_len] = np.append(batchs[i % new_batchs_len], [sample], axis=0)
        batchs.remove(batchs[-1])
    return batchs

        

class NeuralNetwork:
    def __init__(self, layers_lenght):
        self.layers_lenght = layers_lenght
        
        self.initializeWeightsBias()

    def initializeWeightsBias(self):
        layers_lenght = self.layers_lenght

        n_wb = len(layers_lenght) - 1
        w_shapes = [(layers_lenght[i], layers_lenght[i+1]) for i in range(n_wb)]
        b_shapes = [(1, lenght) for lenght in layers_lenght[1:]]

        self.weights = [np.random.rand(shape[0], shape[1]) * 0.1 - 0.05 for shape in w_shapes]
        self.bias = [np.zeros((shape[0], shape[1])) for shape in b_shapes]
        
    def forward(self, x, y, keep_rate=0.85, lambd=0.8, dropout=True):
        
        m = x.shape[0]

        # easy writting
        W = self.weights
        B = self.bias
        n_wb = len(W)
        n_wb_no_y = n_wb - 1

        # forwardpropagation
        A = [x]
        D = [] # dropout masks

        # - hidden layers
        i = 0
        for i in range(n_wb_no_y):
            z = A[i].dot(W[i]) + B[i]
            a = relu(z)

            if dropout:
                d = np.random.rand(a.shape[0], a.shape[1]) < keep_rate
                a *= d
                a /= keep_rate
                D.append(d)
            A.append(a)
        
        # - output layer
        i += 1
        z = A[i].dot(W[i]) + B[i]
        a = softmax(z)
        A.append(a)

        # cost function
        cost = -(1/m) * np.sum(y * np.log(a))
        squared_weights_sum = np.sum([np.sum(np.square(w)) for w in W])
        l2_regularization_cost = (lambd/m) * squared_weights_sum
        # cost += l2_regularization_cost

        cache = {
            'A': A,
            'D': D,
            'predict': A[-1]
        }
        return cost, cache

    def train(self, x, y, epochs=100, keep_rate=0.85, learning_rate=0.03, lambd=0.8, beta=0.9, beta2=0.999, epsilon=10**-8, batch_size=0):
        # mini batching
        batchs_x = batch(x, batch_size)
        batchs_y = batch(y, batch_size)
        batchs_len = len(batchs_x)



        ## BACKPROPAGATION ##
        
        m = x.shape[0]

        # easy writting
        W = self.weights
        B = self.bias
        n_wb = len(W)
        n_wb_no_y = n_wb - 1
        
        # momentum optimizer
        v_dw = [np.zeros(w.shape) for w in W]
        v_db = [np.zeros(b.shape) for b in B]

        # RMSprop optimizer (Root mean square propagation)
        s_dw = [np.zeros(w.shape) for w in W]
        s_db = [np.zeros(b.shape) for b in B]

        cost_list = []
        for epoch in range(epochs):
            b_x = batchs_x[epoch % batchs_len]
            b_y = batchs_y[epoch % batchs_len]

            cost, cache = self.forward(b_x, b_y, keep_rate)
            cost_list.append(cost)
            if epoch*100/epochs  % 5 == 0:
                percent = (epoch * 100) // epochs
                print("Cost: {}, epoch: {}/{} ({}%).".format(cost, epoch, epochs, percent))
            A = cache['A']
            D = cache['D']
            
            ## dCost/dz ##
            dZ = []

            # dCost/dz derivate output layer
            dz = A[-1] - b_y
            dZ.insert(0, dz)

            # dCost/dz hidden layers
            for i in range(n_wb_no_y):
                da = dZ[0].dot(W[-1-i].T)
                da *= D[-1-i]
                da /= keep_rate
                dz = da * drelu(A[-2-i])
                dZ.insert(0, dz)

            # dcost/(dW||dB) and updating weights and biases
            for i in range(n_wb):
                dw = (1/m) * A[i].T.dot(dZ[i]) + (lambd * W[i])/m
                db = (1/m) * np.sum(dZ[i], axis=0, keepdims=True)

                # momentum optimizer equations
                v_dw[i] = beta * v_dw[i] + (1-beta) * dw
                v_db[i] = beta * v_db[i] + (1-beta) * db

                # RMSprop optimizer equations
                s_dw[i] = beta2 * s_dw[i] + (1 - beta2) * dw**2
                s_db[i] = beta2 * s_db[i] + (1 - beta2) * db**2

                # Adam optimizer (derived from adaptive moment estimation)
                self.weights[i] -= v_dw[i] * (learning_rate / (np.sqrt(s_dw[i]) + epsilon))
                self.bias[i] -= v_db[i] * (learning_rate / (np.sqrt(s_db[i]) + epsilon))
        
        return cost_list