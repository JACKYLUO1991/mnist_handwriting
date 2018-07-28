a. 网络图为手工绘制图像，简单的三层感知器
b~c. python MNIST_train_test.py对网络进行训练，同时对验证集（或者说测试集）进行测试
	仅仅训练了20个epoch
	训练过程在logs中
d. python MNIST_inference.py 对图像进行了测试
e. 最优模型保存在了model文件中
    training error=0.00017961419088714732
	test precision=0.972335
	test error=0.02766500000000005
f. 文件转换MNIST_convert.py
	转换成csv文件格式，mnist文件夹下
