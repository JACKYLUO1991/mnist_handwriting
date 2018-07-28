"""
Created by Jacky LUO
Using python3.5
"""


import numpy as np


class DigitClass(object):
    """
    input_nodes: number of input nodes
    hidden_nodes: number of hidden nodes
    output_nodes: number of output nodes
    lr: learning late
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr=0.5):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = lr

        # Initialization weights, Reference: He et.al 2015
        self.w0 = np.random.randn(self.hidden_nodes, self.input_nodes) / np.sqrt(self.input_nodes / 2)
        # self.bias0 = np.zeros(self.hidden_nodes, 1)
        self.w1 = np.random.randn(self.output_nodes, self.hidden_nodes) / np.sqrt(self.hidden_nodes / 2)
        # self.bias1 = np.zeros(self.output_nodes, 1)

        # Activation function(RELU)
        # self.activation_function = lambda x: np.maximum(x, 0)
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))

    def train(self, input_lists, output_lists):
        """
        hidden errors = (weights)T . (errors)--->outputs
        theta Wjk = a * Ek * sigmoid(Ok) * (1 - sigmoid(Ok)) . (Oj)T
        """
        inputs = np.array(input_lists, ndmin=2).T
        targets = np.array(output_lists, ndmin=2).T

        hidden_input = np.dot(self.w0, inputs)
        hidden_output = self.activation_function(hidden_input)

        final_input = np.dot(self.w1, hidden_output)
        final_output = self.activation_function(final_input)

        errors = targets - final_output
        hidden_errors = np.dot(self.w1.T, errors)

        self.w0 += self.lr * np.dot((hidden_errors * hidden_output * (1 - hidden_output)), np.transpose(inputs))
        self.w1 += self.lr * np.dot((errors * final_output * (1 - final_output)), np.transpose(hidden_output))

        return errors

    def inference(self, input_lists):
        inputs = np.array(input_lists, ndmin=2).T

        hidden_input = np.dot(self.w0, inputs)
        hidden_output = self.activation_function(hidden_input)

        final_input = np.dot(self.w1, hidden_output)
        final_output = self.activation_function(final_input)

        return final_output





