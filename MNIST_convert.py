"""
Created by Jacky LUO
Using python3.5
Reference: https://pjreddie.com/projects/mnist-in-csv
"""


def convert_mnist_csv(img_file, label_file, output_file, n):
    imgf = open(img_file, 'rb')
    labelf = open(label_file, 'rb')
    outputf = open(output_file, 'w')

    imgf.read(16)
    labelf.read(8)
    images = []

    for i in range(n):
        image = [ord(labelf.read(1))]
        for j in range(28 * 28):
            image.append(ord(imgf.read(1)))
        images.append(image)

    for image in images:
        outputf.write(",".join(str(pixel) for pixel in image) + "\n")

    imgf.close()
    labelf.close()
    outputf.close()


if __name__ == '__main__':
    convert_mnist_csv("mnist/train-images-idx3-ubyte",
                      "mnist/train-labels-idx1-ubyte",
                      "mnist/mnist_train.csv",
                      60000)

    convert_mnist_csv("mnist/t10k-images-idx3-ubyte",
                      "mnist/t10k-labels-idx1-ubyte",
                      "mnist/mnist_test.csv",
                      10000)
    print('Finished data convert!')








