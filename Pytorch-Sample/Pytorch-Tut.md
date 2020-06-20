# Pytorch Note README

## Tensor opearations 张量/矩阵操作

- Tensor creation

    <Details>
    <summary>创建Tensor</summary>
    1. 可以自动创建随机/全0/全1/`full`(全n)的Tensor，需要给定形状</br>
    2. Tensor可以通过`view`方法，针对当前Tensor的维度结构进行重新构造，并返回一个副本</br>
    3. 可以通过提供另一个Tensor，已定义的随机/全1/全0/`full_like`(全n)方式构建维度相同的新Tensor</br>
    4. 通过调用`torch.from_numpy`方法并传入Numpy的`narray`型数据，可以构建与其维度对应的Tensor</br>
    5. 利用`eye`方法，可以构建单位矩阵，为二维张量</br>
    6. 可以通过定义mean与std来生成所有元素基于指定正态分布的Tensor,mean与std以成对的方式出现，或其中一者为单实数值，一者为Tensor<br/>
    </Details>

    <Details>
    <summary>Tensor的一些性质</summary>
    1. 通过与`Python list`相似的方式切片，如`tensor[3,0]`可取出处在第一维第四个，第二维第一个下维度的全部数据</br>
    2. 相似的，Tensor也可以通过调用`numpy`转化为Numpy中的`ndarray`</br>
    3. 通过`size`和`shape`方法，可以查看当前Tensor的维度状况</br>
    4. Tensor有`narrow`，`transpose`，`unbind`，`split`，`where`等函数操作，分别执行批量指定维度截取(只留下指定部分)，矩阵转置，矩阵裁切(裁切出的各个部分以多个Tensor的方式返回)，矩阵解绑(按指定维度拆为多个Tensor)，矩阵搜索(所有匹配部分用指定Tensor中对应部分替换)</br>
    5. Tensor利用`@`替换乘号标注矩阵相乘，可以调用`add_`方法，参数为合法形状的Tensor或者单个实数，进行矩阵加法</br>
    6. `torch.stack`命令通过处理`list`形式的Tensor集，将多个同样shape的Tensor合并并构成一个整个Tensor，list中的每个Tensor在整合Tensor中依list中顺序在第一维度分布</br>
    </Details>

    [__More Info__](./tensorOptions.py "Tensor Operations")

- Tensor attribution 

    <Details>
    <summary>Tensor的部分常用操作</summary>
    1. 通过利用`Variable`封装Tesnor，并设置参数`requires_grad=True`时，该Tensor若发生变化，其变化率，即Gradient，会被自动计算并作为属性记入当前Tensor</br>
    2. 通过向后传播计算梯度：`tensor.backward()`</br>
    3. 取出当前Tensor的Gradient：`tensor.grad`</br>
    4. 取出当前Tensor中Data：`tensor.data`</br>
    5. 取出当前Tensor的形状：`tensor.shape`</br>
    </Details>

    [__More Info__](./pytorchVariable.py "Tensor Attributions")

## Pytorch NN 部分基础内容

- Activation function 激活函数
  - Sigmoid：
    $$
        f(x) = \frac{1}{1+e^{-x}}
    $$

  - ReLu:
    $$
        f(x) = max(0,x)
    $$

  - Tanh:
    $$
        f(x) = \frac{Sinh(x)}{Cosh(x)} = \frac{e^x-e^{-x}}{e^x+e^{-x}}
    $$

  - SoftPlus:
    $$
        f(x) = log(1+e^x)
    $$

  [__More Info__](./pytorchActivitionFunc.py "Activation Functions")