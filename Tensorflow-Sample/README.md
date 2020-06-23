# Tensorflow Note README

## Tensor foundation 一些TensorFlow框架的基本操作

- TensorFlow使用时的一些基本操作：

  - `tf.enable_eager_execution()`：不完全加载整个框架，只载入利用依赖，贪婪型的执行当前代码，加速小规模测试用代码执行
  
  - `tf.Session()`：TensorFlow框架执行器，控制训练框架的流程逻辑，通过`session.run(tf.global_variables_initializer())`初始化框架中所使用的全局变量状态
  
  - `loss function`，`optimizer`，`training strategy (batch training, etc.)`，`learning rate (decay learning rate, Momemtm, etc.)`，`regularization strategy (ln Norm)`等训练时需要的工具方法均直接使用变量对其进行定义
    
    - TensorFlow中定义的计算时使用变量，只有在被`tf.Session()`中唤醒才会对其根据`feed_dict={}`中给出的参数进行计算，可以理解为Tensorflow对每个需求变量如何计算提前做好定义，只有在Session中该变量被需求，才会真正执行计算 
  
  - 一个简单Full Connect Neural Network构建例子，以及FC层的`weight`与`bias`根据基于正态分布随机初始化方法

- 利用Numpy包执行的一些基本的ndarray压缩操作：
  
  - `numpy.reshape(array, shape)`：转换数组的维度形状，但shape指定为-1时，将整个数组全拆降至一维
  
  - `numpy.squeeze(array, axis=None)`：将指定数组中维度为1的部分删去(由于维度为1的维度直接包含了之后全部数据内容，故视作冗余维度进行删除，如从`[[[1,2],[3,4]]]`可挤压到`[[1,2],[3,4]]`)
  
  - `numpy.flatten(order='C')`此处使用的flatten命令将会返回当亲数组的一份拷贝，对其执行操作并不会影响原矩阵的内容呈现，`order`参数默认为C，表示按行序排列副本，其余参数设置如下：
    
    - C：行序优先
    
    - F：列序优先
    
    - A：依照内存中顺序按行优先排列
    
    - K：依照内存存储顺序排列
  
  - `numpy.ravel(order='C')`与flatten命令不同，ravel命令返回为目标array的识图引用，也就是针对其返回部分切片进行操作会直接更改数据来源的array中的实际数据内容，参数设置和flatten类似
  
- TensorFlow CNN简单示例

  - 整体示例分为：`forward`传播，`backward`传播与`test`三个部分

[__More Info__](./tensorNote.ipynb "TensorFlow basics notes")

## TensorFlow 简单RMB网络构建示例