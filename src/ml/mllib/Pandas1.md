---
title: Pandas入门
type: mllib
order: 2
---

## Pandas介绍
pandas是Python在数据处理方面功能最为强大的扩展模块。在处理实际的金融数据时，一个条数据通常包含了多种类型的数据，例如，股票的代码是字符串，收盘价是浮点型，而成交量是整型等。在C++中可以实现为一个给定结构体作为单元的容器，如向量（vector，C++中的特定数据结构）。在Python中，pandas包含了高级的数据结构Series和DataFrame，使得在Python中处理数据变得非常方便、快速和简单。

### Numpy 和 Pandas 有什么不同
如果用 python 的列表和字典来作比较, 那么可以说 Numpy 是列表形式的，没有数值标签，而 Pandas 就是字典形式。Pandas是基于Numpy构建的，让Numpy为中心的应用变得更加简单。

pandas主要的两个数据结构是Series和DataFrame，随后两节将介绍如何由其他类型的数据结构得到这两种数据结构，或者自行创建这两种数据结构，我们先导入它们以及相关模块：

``` python
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
```

## Pandas数据结构：Series
从一般意义上来讲，Series可以简单地被认为是一维的数组。Series和一维数组最主要的区别在于Series类型具有索引（index）。

### 创建Series

创建一个Series的基本格式是s = Series(data, index=index, name=name)，以下给出几个创建Series的例子。首先我们从数组创建Series：

``` python
In [18]: a = np.random.randn(5)

In [19]: a
Out[19]: array([-0.13907742, -1.13472176, -0.30952444,  0.5945551 , -1.11253943])

In [20]: s = Series(a)

In [21]: s
Out[21]:
0   -0.139077
1   -1.134722
2   -0.309524
3    0.594555
4   -1.112539s = pd.Series([1,3,6,np.nan,44,1])
dtype: float64
```

由于我们没有为数据指定索引。于是会自动创建一个0到N-1（N为长度）的整数型索引。

可以在创建Series时添加index，并可使用Series.index查看具体的index。需要注意的一点是，当从数组创建Series时，若指定index，那么index长度要和data的长度一致：

``` python
In [26]: s = Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

In [27]: s
Out[27]:
a    0.391180
b   -0.909680
c   -0.751830
d   -0.628885
e   -2.225025
dtype: float64

In [28]: s.index
Out[28]: Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
```

创建Series的另一个可选项是name，可指定Series的名称，可用Series.name访问。在随后的DataFrame中，每一列的列名在该列被单独取出来时就成了Series的名称：

``` python
In [29]: s = Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'], name='my_series')

In [30]: s
Out[30]:
a   -0.061299
b    1.448393
c   -2.871637
d   -0.254603
e   -0.970275
Name: my_series, dtype: float64

In [31]: s.name
Out[31]: 'my_series'
```

Series还可以从字典（dict）创建：

``` python
In [35]: d = {'a': 0., 'b': 1, 'c': 2}

In [36]: s = Series(d)

In [37]: s
Out[37]:
a    0.0
b    1.0
c    2.0
dtype: float64
```

让我们来看看使用字典创建Series时指定index的情形（index长度不必和字典相同）：

``` python
In [38]: Series(d, index=['b', 'c', 'd', 'a'])
Out[38]:
b    1.0
c    2.0
d    NaN
a    0.0
dtype: float64
```

我们可以观察到两点：一是字典创建的Series，数据将按index的顺序重新排列；二是index长度可以和字典长度不一致，如果多了的话，pandas将自动为多余的index分配NaN（not a number，pandas中数据缺失的标准记号)，当然index少的话就截取部分的字典内容。

如果数据就是一个单一的变量，如数字4，那么Series将重复这个变量：

``` python
In [39]: Series(4., index=['a', 'b', 'c', 'd', 'e'])
Out[39]:
a    4.0
b    4.0
c    4.0
d    4.0
e    4.0
dtype: float64
```

### Series数据的访问
访问Series数据可以和数组一样使用下标，也可以像字典一样使用索引，还可以使用一些条件过滤：

``` python
In [40]: s = Series(np.random.randn(10),index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    ...: s[0]
    ...:
Out[40]: 1.6327010042454009

In [41]: s[:2]
Out[41]:
a    1.632701
b   -0.808834
dtype: float64
```

``` python
In [42]: s[[2,0,4]]
Out[42]:
c   -1.708988
a    1.632701
e   -0.989622
dtype: float64

In [43]: s[['e', 'i']]
Out[43]:
e   -0.989622
i   -0.901893
dtype: float64
```

``` python
In [44]: s[s > 0.5]
Out[44]:
a    1.632701
d    2.814273
g    1.337555
j    1.314560
dtype: float64

In [45]: 'e' in s
Out[45]: True
```

## Pandas数据结构：DataFrame

在使用DataFrame之前，我们说明一下DataFrame的特性。DataFrame是将数个Series按列合并而成的二维数据结构，每一列单独取出来是一个Series，这和SQL数据库中取出的数据是很类似的。所以，按列对一个DataFrame进行处理更为方便，用户在编程时注意培养按列构建数据的思维。DataFrame的优势在于可以方便地处理不同类型的列，因此，就不要考虑如何对一个全是浮点数的DataFrame求逆之类的问题了，处理这种问题还是把数据存成NumPy的matrix类型比较便利一些。

### 创建DataFrame
首先来看如何从字典创建DataFrame。DataFrame是一个二维的数据结构，是多个Series的集合体。我们先创建一个值是Series的字典，并转换为DataFrame：

``` python
In [46]: d = {'one': Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
    ...: df = DataFrame(d)
    ...:

In [47]: df
Out[47]:
   one  two
a  1.0  1.0
b  2.0  2.0
c  3.0  3.0
d  NaN  4.0
```

可以指定所需的行和列，若字典中不含有对应的元素，则置为NaN：
``` python
In [48]: df = DataFrame(d, index=['r', 'd', 'a'], columns=['two', 'three'])

In [49]: df
Out[49]:
   two three
r  NaN   NaN
d  4.0   NaN
a  1.0   NaN
```

可以使用dataframe.index和dataframe.columns来查看DataFrame的行和列，dataframe.values则以数组的形式返回DataFrame的元素：

``` python
In [50]: df.index
Out[50]: Index(['r', 'd', 'a'], dtype='object')

In [51]: df.columns
Out[51]: Index(['two', 'three'], dtype='object')

In [52]: df.values
Out[52]:
array([[nan, nan],
       [4.0, nan],
       [1.0, nan]], dtype=object)
```

DataFrame也可以从值是数组的字典创建，但是各个数组的长度需要相同：

``` python
In [53]: d = {'one': [1., 2., 3., 4.], 'two': [4., 3., 2., 1.]}

In [54]: df = DataFrame(d, index=['a', 'b', 'c', 'd'])

In [55]: df
Out[55]:
   one  two
a  1.0  4.0
b  2.0  3.0
c  3.0  2.0
d  4.0  1.0
```

值非数组时，没有这一限制，并且缺失值补成NaN：

``` python
In [56]: d= [{'a': 1.6, 'b': 2}, {'a': 3, 'b': 6, 'c': 9}]
    ...: df = DataFrame(d)
    ...:

In [57]: df
Out[57]:
     a  b    c
0  1.6  2  NaN
1  3.0  6  9.0
```

在实际处理数据时，有时需要创建一个空的DataFrame，可以这么做
``` python
In [58]: df = DataFrame()

In [59]: df
Out[59]:
Empty DataFrame
Columns: []
Index: []
```

另一种创建DataFrame的方法十分有用，那就是使用concat函数基于Series或者DataFrame创建一个DataFrame

``` python
In [62]: a = Series(range(5))
    ...: b = Series(np.linspace(4, 20, 5))
    ...: df = pd.concat([a, b], axis=1)
    ...:

In [63]: df
Out[63]:
   0     1
0  0   4.0
1  1   8.0
2  2  12.0
3  3  16.0
4  4  20.0
```

其中的axis=1表示按列进行合并，axis=0表示按行合并，并且，Series都处理成一列，所以这里如果选axis=0的话，将得到一个10×1的DataFrame。下面这个例子展示了如何按行合并DataFrame成一个大的DataFrame：

``` python
In [64]: df = DataFrame()
    ...: index = ['alpha', 'beta', 'gamma', 'delta', 'eta']
    ...: for i in range(5):
    ...:     a = DataFrame([np.linspace(i, 5*i, 5)], index=[index[i]])
    ...:     df = pd.concat([df, a], axis=0)
    ...:

In [65]: df
Out[65]:
         0    1     2     3     4
alpha  0.0  0.0   0.0   0.0   0.0
beta   1.0  2.0   3.0   4.0   5.0
gamma  2.0  4.0   6.0   8.0  10.0
delta  3.0  6.0   9.0  12.0  15.0
eta    4.0  8.0  12.0  16.0  20.0
```

### DataFrame数据的访问
首先，再次强调一下DataFrame是以列作为操作的基础的，全部操作都想象成先从DataFrame里取一列，再从这个Series取元素即可。可以用datafrae.column_name选取列，也可以使用dataframe[]操作选取列，我们可以马上发现前一种方法只能选取一列，而后一种方法可以选择多列。若DataFrame没有列名，[]可以使用非负整数，也就是“下标”选取列；若有列名，则必须使用列名选取，另外datafrae.column_name在没有列名的时候是无效的：

``` python
In [66]: df[1]
Out[66]:
alpha    0.0
beta     2.0
gamma    4.0
delta    6.0
eta      8.0
Name: 1, dtype: float64

In [67]: type(df[1])
Out[67]: pandas.core.series.Series
```

``` python
In [71]: df.columns = ['a', 'b', 'c', 'd', 'e']

In [72]: df['b']
Out[72]:
alpha    0.0
beta     2.0
gamma    4.0
delta    6.0
eta      8.0
Name: b, dtype: float64

In [78]: df.b
Out[78]:
alpha    0.0
beta     2.0
gamma    4.0
delta    6.0
eta      8.0
Name: b, dtype: float64

In [79]: df[['a','b']]
Out[79]:
         a    b
alpha  0.0  0.0
beta   1.0  2.0
gamma  2.0  4.0
delta  3.0  6.0
eta    4.0  8.0
```

以上代码使用了dataframe.columns为DataFrame赋列名，并且我们看到单独取一列出来，其数据结构显示的是Series，取两列及两列以上的结果仍然是DataFrame。访问特定的元素可以如Series一样使用下标或者是索引:

``` python
In [84]: df.b[2]
Out[84]: 4.0

In [85]: df.b.gamma
Out[85]: 4.0
```

若需要选取行，可以使用dataframe.iloc按下标选取，或者使用dataframe.loc按索引选取：

``` python
In [86]: df.iloc[1]
Out[86]:
a    1.0
b    2.0
c    3.0
d    4.0
e    5.0
Name: beta, dtype: float64

In [87]: df.loc['beta']
Out[87]:
a    1.0
b    2.0
c    3.0
d    4.0
e    5.0
Name: beta, dtype: float64
```

选取行还可以使用切片的方式或者是布尔类型的向量：

``` python
In [88]: df[1:3]
Out[88]:
         a    b    c    d     e
beta   1.0  2.0  3.0  4.0   5.0
gamma  2.0  4.0  6.0  8.0  10.0

In [89]: bool_vec = [True, False, True, True, False]

In [90]: df[bool_vec]
Out[90]:
         a    b    c     d     e
alpha  0.0  0.0  0.0   0.0   0.0
gamma  2.0  4.0  6.0   8.0  10.0
delta  3.0  6.0  9.0  12.0  15.0
```

行列组合起来选取数据：

``` python
In [91]: df[['b', 'd']].iloc[[1, 3]]
Out[91]:
         b     d
beta   2.0   4.0
delta  6.0  12.0

In [92]: df.iloc[[1, 3]][['b', 'd']]
Out[92]:
         b     d
beta   2.0   4.0
delta  6.0  12.0

In [93]: df[['b', 'd']].loc[['beta', 'delta']]
Out[93]:
         b     d
beta   2.0   4.0
delta  6.0  12.0
```

如果不是需要访问特定行列，而只是某个特殊位置的元素的话，dataframe.at和dataframe.iat是最快的方式，它们分别用于使用索引和下标进行访问：

``` python
In [94]: df.iat[2, 3]
Out[94]: 8.0

In [95]: df.at['gamma', 'd']
Out[95]: 8.0
```

dataframe.ix可以混合使用索引和下标进行访问，唯一需要注意的地方是行列内部需要一致，不可以同时使用索引和标签访问行或者列，不然的话，将会得到意外的结果：

``` python
In [97]: df.ix[[1, 2], ['b', 'e']]
Out[97]:
         b     e
beta   2.0   5.0
gamma  4.0  10.0

In [98]: df.ix[[1, 2], ['b', 4]]
Out[98]:
         b   4
beta   2.0 NaN
gamma  4.0 NaN
```