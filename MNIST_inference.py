"""
Created by Jacky LUO
Using python3.5
"""

from MNIST_net import DigitClass
import numpy as np
import os
from scipy import misc
import glob


input_nodes = 784
hidden_nodes = 128
output_nodes = 10

mnist = DigitClass(input_nodes, hidden_nodes, output_nodes)

matrix = np.load("model/epoch_19_mnist.npy")
mnist.w0 = matrix[:, :784 * 128].reshape(128, 784)
mnist.w1 = matrix[:, 784 * 128:].reshape(10, 128)

for filename in glob.glob("image/*.png"):
    im = misc.imread(filename, 'L')
    label = np.argmax(mnist.inference(np.asfarray(im.reshape(1, 784) / 255.0)))
    print("The label is: ", label)
