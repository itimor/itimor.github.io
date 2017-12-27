---
title: 深入探索MNIST
type: ml
order: 3
---

## 写在前面
TensorFlow是一个非常强大的用来做大规模数值计算的库。其所擅长的任务之一就是实现以及训练深度神经网络。

在本教程中，我们将学到构建一个TensorFlow模型的基本步骤，并将通过这些步骤为MNIST构建一个深度卷积神经网络。

## 安装
在创建模型之前，我们会先加载MNIST数据集，然后启动一个TensorFlow的session。

### 加载MNIST数据

为了方便起见，我们已经准备了[一个脚本](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py)来自动下载和导入MNIST数据集。它会自动创建一个 `'MNIST_data'` 的目录来存储数据。

``` python
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

这里，`mnist` 是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。同时提供了一个函数，用于在迭代中获得 `minibatch` ，后面我们将会用到。

### 运行TensorFlow的InteractiveSession

Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。