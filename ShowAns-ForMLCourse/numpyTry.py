import numpy as np


def spiltLine():
    print("<===============这里是分割线===============>")

def npArray():
    spiltLine()
    aList = np.array([2,3,4,5,6])
    bList = np.array([3,2,2,4,3])    
    aList = aList ** bList
    print("Array的数据批量处理：",aList)

    spiltLine()
    cList = np.array([[3,4,5,2],[7,9,3,6],[5,4,3,8]])
    print("Array的相关属性\n秩: ",cList.ndim,
    "\n尺度/大小: ",cList.shape,
    "\n对象个数: ",cList.size,
    "\n元素类型: ",cList.dtype,
    "\n每个元素大小: ",cList.itemsize)

    spiltLine()
    dList = np.ones((2,4,3))
    print("创建展示1 ：",dList)
    print("创建展示2 ：",np.full_like(cList,6))
    print("创建展示3 ：",np.linspace(1,9,4,endpoint=False))
    print("创建展示4 ：",np.concatenate((aList,aList,bList)))

    spiltLine()
    print("不改变reshape :",dList.reshape((4,6)),
    "\n原数组：",dList)
    # 转置矩阵
    print("交换swapaxes :",cList.swapaxes(0,1))
    # 降低维度
    print("降维flatten :",cList.flatten())
    print("变换类型 :",dList.astype(np.float))
    

if __name__ == "__main__":
    npArray()
