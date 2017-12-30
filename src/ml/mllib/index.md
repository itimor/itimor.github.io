---
title: Numpy入门
type: mllib
order: 1
---

## Numpy介绍

Numpy是Python的一个开源科学计算的库，提供了矩阵运算的功能，其一般与Scipy、matplotlib一起使用。

NumPy提供了两种基本的对象：`ndarray`（N-dimensional array object）和 `ufunc`（universal function object）。ndarray(下文统一称之为数组)是存储单一数据类型的多维数组，而ufunc则是能够对数组进行处理的函数。

**环境说明**： python3.6

### 安装
``` python
pip install numpy
```
### 测试
``` python
import numpy as np
print(np.version.version)
1.13.3
```

### ndarray
ndarray是一个多维数组对象，由两部分构成：

> 实际的数据

> 描述这些数据的元数据（数据维度、数据类型等）

ndarray数组一般要求所有元素类型相同（同质），数组下标从0开始;

在NumPy中数组的维度(dimensions)叫做轴(axis)，轴的个数叫做秩(rank)。

## 创建ndarray

* 从Python中的列表、元组等类型创建ndarray数组
　　

可以使用np.array()方法将Python列表或元组转化为数组，转化后的数组元素的类型由原来的对象的类型来决定。


（1）一维数组
``` python
>>> import numpy as np
>>> np.array((1, 2, 3, 4))  #从元组创建
array([1, 2, 3, 4])
```

（2）多维数组
``` python
>>> np.array([[1, 2, 3, 4],[5, 6, 7, 8]])   #从列表创建
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
>>> np.array([[1, 2, 3, 4],(5, 6, 7, 8)])   #从列表和元组混合类型创建
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
```

当np.array()不指定dtype时，NumPy将根据数据情况关联一个dtype类型。
* 使用NumPy中函数创建ndarray数组，如：arange, ones, zeros等


函数	| 说明
--- | ---
np.arange(n)	|类似range()函数，返回ndarray类型，元素从0到n‐1
np.ones(shape)	|根据shape生成一个全1数组，shape是元组类型
np.zeros(shape)	|根据shape生成一个全0数组，shape是元组类型
np.full(shape,val)	|根据shape生成一个数组，每个元素值都是val
np.eye(n)	|创建一个正方的n*n单位矩阵，对角线为1，其余为0
np.ones_like(a)	|根据数组a的形状生成一个全1数组
np.zeros_like(a)	|根据数组a的形状生成一个全0数组
np.full_like(a,val)	|根据数组a的形状生成一个数组，每个元素值都是val
np.linspace()	|根据起止数据等间距地填充数据，形成数组
np.concatenate()	|将两个或多个数组合并成一个新的数组

``` python
>>> import numpy as np
>>> np.arange(6)
array([0, 1, 2, 3, 4, 5])
>>> np.ones(6)
array([ 1.,  1.,  1.,  1.,  1.,  1.])
>>> np.zeros((3,2))
array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]])
>>> np.full((2,2),6)
array([[6, 6],
       [6, 6]])
>>> np.eye(3)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> a = np.array([[1,2],[2,1]])
>>> np.ones_like(a) #生成一个与a数组相同结构的全1数组
array([[1, 1],
       [1, 1]])
>>> np.linspace(1, 10, 6)   # 1为起始数，10为终止数，6为数的个数
array([  1. ,   2.8,   4.6,   6.4,   8.2,  10. ])
```

## ndarray的属性

**shape**：一个说明ndarray对象各维度大小的元组；对一个n 行m 列的矩阵来说，shape为 (n,m)。

``` python
>>> import numpy as np
>>> c = np.array([1, 2, 3, 4])
>>> c.shape
(4,)
>>> c.shape = 2, 2
>>> c
array([[1, 2]
       [3, 4]])
>>> k = c.reshape(4,)
>>> k
array([1, 2, 3, 4])
>>> k[0] = 0
>>> k
array([0, 2, 3, 4])
>>> c
array([[0, 2]
       [3, 4]])
```


> 使用数组的reshape方法，可以创建一个改变了尺寸的新数组，原数组的shape保持不变。

>k与c共享内存区域，所以修改其中任意一个数组的元素都会同时修改另外一个数组的内容。


**dtype**：ndarray对象的元素类型;数组的元素类型可以通过dtype属性获得，可以通过dtype参数在创建时指定元素类型。类型可以是 numpy.int32, numpy.int16, and numpy.float64 等。

``` python
>>> c = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
>>> c.dtype
dtype('int32')
```
创建数组时也可以指定元素的数据类型:
``` python
>>>np.array([[1,2,3,4],[5, 6, 7, 8]], dtype = np.float)
array([[ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.]])
```

**ndim**：数组的维数，也称为rank(秩)
``` python
>>> c = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
>>> c.ndim
2
```

**size**：ndarray对象元素的个数
``` python
>>> c = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
>>> c.size
8
```

**itemsize**：ndarray对象中每个元素的大小，即占用的字节数。
``` python
>>> c = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
>>> c.itemsize
4
```

**data**：指向数据内存。
``` python
>>> c = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
>>> c.data
<read-write buffer for 0x02D89E30, size 32, offset 0 at 0x02CE7DE0>
```

## ndarray的数据类型
Python语法仅支持整数、浮点数和复数3种类型，而ndarray支持多种数据类型，这也正是其用于科学计算的强大之处。


类型	|说明
---|---
bool	|布尔类型，True或False
intc	|与C语言中的int类型一致，一般是int32或int64
intp	|用于索引的整数，与C语言中ssize_t一致，int32或int64
int8	|字节长度的整数，取值：[‐128, 127]
int16	|16位长度的整数，取值：[‐32768, 32767]
int32	|32位长度的整数，取值：[‐2^31,2^31‐1]
int64	|64位长度的整数，取值：[‐2^63,2^63‐1]
uint8	|8位无符号整数，取值：[0, 255]
uint16	|16位无符号整数，取值：[0, 65535]
uint32	|32位无符号整数，取值：[0,2^32‐1]
uint64	|32位无符号整数，取值：[0,2^64‐1]
float16	|16位半精度浮点数：1位符号位，5位指数，10位尾数
float32	|32位半精度浮点数：1位符号位，8位指数，23位尾数
float64	|64位半精度浮点数：1位符号位，11位指数，52位尾数
complex64	|复数类型，实部和虚部都是32位浮点数
complex128	|复数类型，实部和虚部都是64位浮点数


## ndarray数组的操作
### 索引和切片

* 索引：获取数组中特定位置元素的过程

* 切片：获取数组元素子集的过程


一维数组的索引和切片
``` python
>>> a = np.array([1, 2, 3, 4, 5, 6])
>>> a[2]
3
>>> a[-2]   # 从右开始
5
>>> a[1:4]      # 起始编号:终止编号(不含):步长
array([2, 3, 4])
>>> a[0:6:2]    # 指定步长为2
array([1, 3, 5])
```

多维数组的索引和切片
``` python
>>> a = np.array([[1, 2, 3],[4, 5, 6]])
>>> a
array([[1, 2, 3],
       [4, 5, 6]])
>>> a[1, 2]
6
>>> a[-1,-2]    # 从右开始
5
# 选取一个维度用:，不同维度用逗号分开
>>> a[:1, 1:]   # 每个维度切片方法与一维数组相同
array([[2, 3]])
```

### 基本运算
* 一元函数

函数	|说明
---|---
np.abs(x) np.fabs(x)	|计算数组各元素的绝对值
np.sqrt(x)	|计算数组各元素的平方根
np.square(x)	|计算数组各元素的平方
np.log(x) np.log10(x) np.log2(x)	|计算数组各元素的自然对数、10底对数和2底对数
np.ceil(x) np.floor(x)	|计算数组各元素的ceiling值 或 floor值


``` python
>>> a = np.array([-2, 4, 6])
>>> np.abs(a)
array([2, 4, 6])
>>> a
array([-2,  4,  6])
>>> np.fabs(a)
array([ 2.,  4.,  6.])
>>> b = np.square(a)
>>> b
array([ 4, 16, 36])
>>> np.sqrt(b)
array([ 2.,  4.,  6.])
>>> np.log2(b)
array([ 2.      ,  4.      ,  5.169925])
>>> a = np.array([0.2, 1.5, 2.8])
>>> np.ceil(a)
array([ 1.,  2.,  3.])
>>> np.floor(a)
array([ 0.,  1.,  2.])
```


函数	|说明
---|---
np.rint(x)	|计算数组各元素的四舍五入值
np.modf(x)	|将数组各元素的小数和整数部分以两个独立数组形式返回
np.cos(x) np.cosh(x)	|计算数组各元素的普通型和双曲型三角函数
np.sin(x) np.sinh(x)	|计算数组各元素的普通型和双曲型三角函数
np.tan(x) np.tanh(x)	|计算数组各元素的普通型和双曲型三角函数
np.exp(x)	|计算数组各元素的指数值
np.sign(x)	|计算数组各元素的符号值，1(+),0,‐1(‐)

* 二元函数


函数	|说明
---|---
+, ‐, , /, *	|两个数组各元素进行对应运算
np.maximum(x,y) np.fmax()	|元素级的最大值/最小值计算
np.minimum(x,y) np.fmin()	|元素级的最大值/最小值计算
np.mod(x,y)	|元素级的模运算
np.copysign(x,y)	|将数组y中各元素值的符号赋值给数组x对应元素
\>, <, >=, <=, ==, != |算术比较，产生布尔型数组

``` python
>>> a = np.array([[1, 2, 3],[4, 5, 6]])
>>> b = np.array([[1, 2, 3],[4, 5, 6]])
>>> a + b
array([[ 2,  4,  6],
       [ 8, 10, 12]])
>>> a - b
array([[0, 0, 0],
       [0, 0, 0]])
>>> a * b
array([[ 1,  4,  9],
       [16, 25, 36]])
>>> a / b
array([[1, 1, 1],
       [1, 1, 1]])
>>> a**2
array([[ 1,  4,  9],
       [16, 25, 36]])
>>> np.maximum(a, b)
array([[1, 2, 3],
       [4, 5, 6]])
>>> np.fmax(a, b)
array([[1, 2, 3],
       [4, 5, 6]])
>>> np.mod(a, b)
array([[0, 0, 0],
       [0, 0, 0]])
>>> a >= b
array([[ True,  True,  True],
       [ True,  True,  True]], dtype=bool)

```

## ndarray数组的变换
### 维度变换

方法	|说明
---|---
.reshape(shape)	|不改变数组元素，返回一个shape形状的数组，原数组不变
.resize(shape)	|与.reshape()功能一致，但修改原数组
.swapaxes(ax1,ax2)	|将数组n个维度中两个维度进行调换
.flatten()	|对数组进行降维，返回折叠后的一维数组，原数组不变

``` python
>>> import numpy as np
>>> a = np.ones((2,6))
>>> a
array([[ 1.,  1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.,  1.]])
>>> a.reshape((3,4))
array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]])
>>> a
array([[ 1.,  1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.,  1.]])
>>> a.resize((3,4))
>>> a
array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]])
>>> a.flatten()
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
>>> a
array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]])
```

### 类型变换
astype()方法一定会创建新的数组（原始数据的一个拷贝），即使两个类型一致。

``` python
>>> a = np.ones((2,3), dtype=np.int32)
>>> a
array([[1, 1, 1],
       [1, 1, 1]])
>>> a.astype(np.float)
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> a
array([[1, 1, 1],
       [1, 1, 1]])
```

### 转换成列表
tolist()方式可以实现将ndarray数组转换成列表。

``` python
>>> a = np.ones((2,3), dtype=np.int32)
>>> a
array([[1, 1, 1],
       [1, 1, 1]])
>>> a.tolist()
[[1, 1, 1], [1, 1, 1]]
```
