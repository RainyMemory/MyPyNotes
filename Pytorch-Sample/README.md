# Pytorch Note README

## Tensor opearations 张量/矩阵操作

- Tensor creation

    1. 可以自动创建随机/全0/全1/`full`(全n)的Tensor，需要给定形状</br>
   
    2. Tensor可以通过`view`方法，针对当前Tensor的维度结构进行重新构造，并返回一个副本</br>
   
    3. 可以通过提供另一个Tensor，已定义的随机/全1/全0/`full_like`(全n)方式构建维度相同的新Tensor</br>
   
    4. 通过调用`torch.from_numpy`方法并传入Numpy的`narray`型数据，可以构建与其维度对应的Tensor</br>
   
    5. 利用`eye`方法，可以构建单位矩阵，为二维张量</br>
   
    6. 可以通过定义mean与std来生成所有元素基于指定正态分布的Tensor,mean与std以成对的方式出现，或其中一者为单实数值，一者为Tensor<br/>
   
    7. 通过与`Python list`相似的方式切片，如`tensor[3,0]`可取出处在第一维第四个，第二维第一个下维度的全部数据</br>
   
    8. 相似的，Tensor也可以通过调用`numpy`转化为Numpy中的`ndarray`</br>
   
    9.  通过`size`和`shape`方法，可以查看当前Tensor的维度状况</br>
   
    10. Tensor有`narrow`，`transpose`，`unbind`，`split`，`where`等函数操作，分别执行批量指定维度截取(只留下指定部分)，矩阵转置，矩阵裁切(裁切出的各个部分以多个Tensor的方式返回)，矩阵解绑(按指定维度拆为多个Tensor)，矩阵搜索(所有匹配部分用指定Tensor中对应部分替换)</br>
   
    11. Tensor利用`@`替换乘号标注矩阵相乘，可以调用`add_`方法，参数为合法形状的Tensor或者单个实数，进行矩阵加法</br>
   
    12. `torch.stack`命令通过处理`list`形式的Tensor集，将多个同样shape的Tensor合并并构成一个整个Tensor，list中的每个Tensor在整合Tensor中依list中顺序在第一维度分布</br>

    [__More Info__](./tensorOptions.py "Tensor Operations")

- Tensor attribution 

    1. 通过利用`Variable`封装Tesnor，并设置参数`requires_grad=True`时，该Tensor若发生变化，其变化率，即Gradient，会被自动计算并作为属性记入当前Tensor</br>
   
    2. 通过向后传播计算梯度：`tensor.backward()`</br>

    3. 取出当前Tensor的Gradient：`tensor.grad`</br>
   
    4. 取出当前Tensor中Data：`tensor.data`</br>
   
    5. 取出当前Tensor的形状：`tensor.shape`</br>

    [__More Info__](./pytorchVariable.py "Tensor Attributions")

## Pytorch NN 部分基础内容

- Activation function 激活函数
  
  - Sigmoid：&nbsp; $f(x) = \frac{1}{1+e^{-x}}$

  - ReLu: &nbsp; $f(x) = max(0,x)$

  - Tanh: &nbsp; $f(x) = \frac{Sinh(x)}{Cosh(x)} = \frac{e^x-e^{-x}}{e^x+e^{-x}}$

  - SoftPlus: &nbsp; $f(x) = log(1+e^x)$

  [__More Info__](./pytorchActivitionFunc.py "Activation Functions")

## Pytorch training tricks 训练网络模型时一些常用技巧

- Mini Batch training 批量学习

  - 通过`Data.DataLoader`进行Batch批量配置，包括指定数据源，Batch大小，多线程模式以及是否使用随机采样

  - 利用`enumerate(dataLoader)`对每次Batch中的batch_sample与batch_label进行读取，可集成于for语句中使逻辑清晰

  [__More info__](./pytorchMiniBatch.py "Mini batch training")

- Model save and load 模型参数冻结与重载

  - 直接使用`torch.save(myModel, './~.pkl')`，对模型整体(包含模型结构)进行储存

  - 此时在读取时，我们无需指定网络结构，因在存储时已经将其存入了模型文件中，直接读取使用即可：`torch.load('~.pkl')`

  - 只存储模型参数，是在预训练，早停等策略中的常见手段，此种方法减少了存储的开销并提高了模型的存写效率，可以提高大规模模型的训练效率：`torch.save(myModel.state_dict(), '~.pkl')`
  
  - 此时读取模型，需要与写出文件中参数结构相符的网络结构定义，大型网络模型一般单独列出文件设计保存，若为小型模型，可利用`torch.nn.Sequential()`进行临时定义，之后使用`myModel.load_state_dict('~.pkl')`导入已经定义好的myModel模型中
  
  [__More Info__](./moudleSaveRestore.py "Save & Load model")

- Cuda 激活GPU训练

  - 将数据集和模型参数通过`cuda()`载入GPU，使Tensor计算享受GPU算力加速

  [__More Info__](./pytorchCUDAGuide.md "Load data into GPU")

## Pytorch NN construct tricks 一些模型构建时的常用策略

- Normal distribution parameters 网络层参数依照正态分布初始化策略

  - 使用`init.normal(targetLayer.weight, mean=0, std=.1)`即可对当前指定的targetLayer进行权重生成，将其weight参数按照给出的高斯分布随即采样生成，或直接使用`targetLayer.weight.data.mormal(mean, std)`初始化

  - 使用`init.constant(targetLayer.bias, Bias)`定义该层的bias偏移量，当然，很多情况下我们会希望直接使用无偏网络层进行数据拟合

  [__More Info__](./pytorchBatchNormSample.py "Batch normalization & layer weight init")

- Batch Normalization 网络指定层输出内容规范化策略

  - 利用`nn.BatchNorm1d()`等函数进行BatchNorm层构建，目的是将获得的Tensor中的数据进行Normalization，可以对一个批次的多条Tensor数据进行一并处理，其输入维度需要与被Normalization的输出层输出维度相符合

  [__More Info__](./pytorchBatchNormSample.py "Batch normalization & layer weight init")
  
- Dropout 网络结构复杂度限制策略

  - 透过在Dropout目标层后定义`torch.nn.Dropout(dropOutRate)`来对该层*训练*时的随机Dropout概率进行管理

  - 在对采用Dropout策略的网络进行测试和使用时，我们需要关闭网络中的Dropout策略来使网络获得更强泛化性能：`myModel.eval()`

## Pytorch NN samples 常见的一些NN网络结构简单例子

- [NN - 全连接FC分类网络](./pytorchNN.py "Full connect classification nerual network sample")

- [NN - 全链接FC回归网络](./pytorchRegsample.py "Full connect regression nerual network sample")

  - Hint：最后输出时可通过Sigmoid(单对象)或Softmax(多对象)进行概率/风险推断导出

- [CNN - 简单卷积网络](./pytorchCNNsample.py "CNN")

- [RNN - 简单LSTM网络](./pytorchRNNsample.py "RNN")

- [GAN - 简单对抗网络](./PytorchGANSample.py "GAN")

- [AutoEncoder & AutoDecoder](./pytorchAutoEncoderSample.py "Encoder")

- [Reinforce Learning - 简单强化学习样例](./pytorchReinforceSample.py "Reinforce Learning")