# Numpy 使用随笔

## **ndArray 部分**
*（此列表中不要求元素类型全部相同，开始下标为0）*

### **ndArray 的各项属性**
- axis 轴 ：数据的维度 array.ndim
- rank 秩 ：轴的数量 array.ndim
- 尺度/大小 ：数据维度，n行m列 array.shape （m仅在其格式符合矩阵时输出）
- 对象个数 ：共多少个数据，矩阵中为m*n，否则输出m array.size
- 元素类型 ：其中数据为何种类型 array.dtype
- 每个元素大小 ：每个数据大小，显示为字节数目 array.itemsize

### **ndArray 特殊创建**
- 使用Python的List或元组进行初始化 np.array(Object)
- np.arrange(n) 内容为从0至n-1
- np.ones(shape) 全1数组，shape为元组类型
- np.zeros(shape) 全0数组，要求同上
- np.full(shape,val) 全val数组，要求同上
- np.eye(n) 返回n*n矩阵，为单位矩阵（对角线为1其余为0）
- np.full_like(List,val) 按照List的shape创建全val数组
- np.ones_like(List) 同上
- np.zeros_like(List) 同上
- np.linspace() 根据起始截止值填充数据给出对应数组（浮点型），第三个参数为生成数量，endpoint参数，用于指定最后一个数字是否必须出现在生成的数组里。
- np.concatenate() 两个或多个数组合并为一个，全放入一个元组中传入

### **ndArray 变换**
- array.reshape(shape) 返回shape形数组，原数组不变
- array.resize(shape) 同上，但是原数组改变
- array.swapaxes(axis1,axis2) 交换数组中两个维度（位置对调）
- array.flatten(Lsit) 将指定的数组投影到一维数组
- list.astype(numpy.type) 变换数组到指定类型，返回一个创建的新数组
- list.tolist() 变换为Python内置列表性变量
- array的切片方式与Python基本相同

### **ndArray numpy部分运算操作**
- mean 均值
- abs/fabs 绝对值
- sqrt 平方根
- square 平方
- log/log2/log10 对数运算（e,2,10）
- ceil/floor 向上/下取整
- rint 四舍五入
- cos/sin/tan/cosh/sinh/tanh 普通与双曲型三角函数
- exp 指数值
- sign 正为1，负为-1，0为0
**（注：基本作用于一元数组）**
- copysign(x,y) y中符号赋予x中成员数字

## *数据的存读处理*

### *.csv 文件的存于读*
- 存：<br>
    np.savetext(frame,array,fmt='%.18e',delimiter=None)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;frame : 文件字符串或产生器，可以为.gz或.bz2<br>
    &nbsp;&nbsp;&nbsp;&nbsp;array : 存入的数据<br>
    &nbsp;&nbsp;&nbsp;&nbsp;fmt : format，存入数据的格式，可以，如%d，%.2f，%.18e，%s<br>
    &nbsp;&nbsp;&nbsp;&nbsp;delimiter : 分隔符，默认空格<br>
- 读：<br>
    np.loadtext(frame,dtype=np.float,delimiter=None,unpack=False)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;unpack : 为True时读入属性将写入不同变量<br>
    
*（csv仅能有效的处理一维，二维数据）*

### *多维数组 文件的存取*
- Array.tofile(frame,sep='',format='%s')<br>
    &nbsp;&nbsp;&nbsp;&nbsp;sep不指定时可写入为二进制文件
- numpy.fromfile(frame,dtype=float,count=-1,sep='')<br>
    &nbsp;&nbsp;&nbsp;&nbsp;dtype不指定则默认为float类型
    &nbsp;&nbsp;&nbsp;&nbsp;count制定读入数量，-1时为读取全部
```
    写入读入后数组的维度信息丢失。
    所以需要使用reshape方式变换为之前维度。
    故使用本方法需要知道写入数组的原先类型与维度信息。
    则通用做法为：
        使用同名csv文件保存对应的维度和类型信息
        利用此信息对读入的二进制文件信息进行变换处理
```

- numpy.save(fname,array)/numpy.savez(fname,array)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;写入为.npz/.npy类型文件
- numpy.load(fname)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;读入numpy格式的数据文件(.npz/.npy)
```
    本方法是可以直接还原出多维数组的原本格式，但文件格式相对固定。
```


### *numpy对数组进行的统计数据处理*
- numpy.sun(List)<br>&nbsp;&nbsp;&nbsp;&nbsp;全体求和
- numpy.mean(List,axis=0/1)<br>&nbsp;&nbsp;&nbsp;&nbsp;横/纵维度求平均值
- numpy.average(List,axis=,weight=[])<br>&nbsp;&nbsp;&nbsp;&nbsp;加权平均
- numpy.std(List)<br>&nbsp;&nbsp;&nbsp;&nbsp;标准差
- numpy.var(List)<br>&nbsp;&nbsp;&nbsp;&nbsp;方差
- numpy.min(List);numpy.max(List)<br>&nbsp;&nbsp;&nbsp;&nbsp;最大最小值
- numpy.ptp(List);numpy.median(List)<br>&nbsp;&nbsp;&nbsp;&nbsp;最大最小值差，中位数
- numpy.argmin(List);numpy.argmax(List)<br>&nbsp;&nbsp;&nbsp;&nbsp;最大最小值降一维下标
- numpy.unravel_index(index,List.shape)<br>&nbsp;&nbsp;&nbsp;&nbsp;一维下表多维化
- numpy.gradient(List)<br>&nbsp;&nbsp;&nbsp;&nbsp;计算每个维度对应的梯度值（变化率/斜率）
- 