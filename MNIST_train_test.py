"""
Created by Jacky LUO
Using python3.5
"""

from MNIST_net import DigitClass
import numpy as np


input_nodes = 28 * 28
hidden_nodes = 128
output_nodes = 10

learning_rate = 0.1
epochs = 20

mnist = DigitClass(input_nodes, hidden_nodes, output_nodes, learning_rate)

with open("mnist/mnist_train.csv", "r") as f:
    training_lists = f.readlines()

with open("mnist/mnist_test.csv", "r") as f:
    test_lists = f.readlines()

score_map = []
min_value = 1
w0_min = 1
w1_min = 1
best_epoch = 0

for epoch in range(epochs):
    print("--------------------- Epoch: %d ---------------------" % epoch)
    for list0 in training_lists:
        errorList = []
        list_no_comma0 = list0.split(",")
        inputs0 = (np.asfarray(list_no_comma0[1:]) / 255.0)
        targets = np.zeros(output_nodes)
        targets[int(list_no_comma0[0])] = 1

        # In order to easy displaying
        error = np.abs(mnist.train(inputs0, targets))
        errorList.append(error)
    print("Training error: ", np.mean(errorList))

    # Python2 using errorList = None
    errorList.clear()

    for list1 in test_lists:
        list_no_comma1 = list1.split(",")
        correct_label = int(list_no_comma1[0])
        inputs1 = (np.asfarray(list_no_comma1[1:]) / 255.0)

        outputs = mnist.inference(inputs1)

        label = np.argmax(outputs)

        if label == correct_label:
            score_map.append(1)
        else:
            score_map.append(0)

    test_error = 1 - np.asarray(score_map).sum() / np.asarray(score_map).size
    test_precision = np.asarray(score_map).sum() / np.asarray(score_map).size

    if test_error < min_value:
        min_value = test_error
        w0_min = mnist.w0
        w1_min = mnist.w1
        best_epoch = epoch

    print("Test Precision: ", test_precision)
    print("Test error: ", test_error)

# Save the weights of the smallest testing error
np.save("model/epoch_{}_mnist.npy".format(best_epoch), np.concatenate([np.asarray(w0_min).reshape(1, 784 * 128),
                                                                       np.asarray(w1_min).reshape(1, 128 * 10)],
                                                                      axis=-1))
print('W0 shape: ', w0_min.shape)
print('W1 shape: ', w1_min.shape)
