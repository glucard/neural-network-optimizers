import numpy as np
import neuralnetwork as NN
from matplotlib import pyplot as plt
from PIL import Image

np.set_printoptions(suppress=True)

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def showLayers(forward_cache):
    A = forward_cache['A']
    shapes = [np.sqrt(a.shape[1]) for a in A]
    
    for i in range(len(A[0])):
        number_prediction = np.argmax(A[-1][i])
        for j in range(len(A)):
            if shapes[j].is_integer():
                shape = int(shapes[j])
                plt.imshow(A[j][i].reshape((shape,shape)), interpolation='nearest')
            else:
                plt.imshow(A[j][i].reshape((1, len(A[j][i]))), interpolation='nearest')
            plt.savefig('predictions/{}_{}_{}'.format(number_prediction, i, j))

def saveLayersImg(forward_cache):
    max_size = 200
    A = forward_cache['A']
    shapes = [np.sqrt(a.shape[1]) for a in A]
    
    for i in range(len(A[0])//10):
        number_prediction = np.argmax(A[-1][i])
        for j in range(len(A)):
            a = A[j][i]
            a = normalizeData(a) * 255
            if shapes[j].is_integer():
                size = max_size, max_size
                shape = int(shapes[j])
                img = a.reshape((shape,shape))
            else:
                size = max_size, max_size // len(a)
                img = a.reshape((1, len(a)))

            img = Image.fromarray(img).convert('RGB')
            img = img.resize(size, Image.NEAREST)
            img.save("predictions_mini_batch_AdamOpitimizer/{}_{}_{}.jpeg".format(number_prediction, i, j))

def showCost(cost_list, cost):
    x = range(len(cost_list))
    plt.plot(x, cost_list)
    plt.savefig('predictions_mini_batch_AdamOpitimizer/_cost_function_graph_{}.png'.format(cost))

def main():
    x = np.loadtxt('inputs.txt')
    y = np.loadtxt('outputs.txt')

    nn = NN.NeuralNetwork([784, 100, 64, 36, 25, 10])

    cost_list = nn.train(x, y, 20000, learning_rate=0.0002, lambd=0.85, batch_size=250)

    cost, forward_cache = nn.forward(x, y, dropout=False)

    showCost(cost_list, cost)
    saveLayersImg(forward_cache)
    

if __name__ == '__main__':
    main()