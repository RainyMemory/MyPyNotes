# Matplotlib 使用随笔

## **pyplot 基础部分**
- pyplot.xlabel/ylabel()&nbsp;&nbsp; 设置图像x,y轴名称
- pyplot.savefig("name", dpi=)&nbsp;&nbsp; png格式按照给出dpi保存图像
- pyplot.plot(x,y,format_string,**kwargs)<br>&nbsp;&nbsp;&nbsp;&nbsp; 参数单List时，默认为y轴，x轴步长为1递增<br>&nbsp;&nbsp;&nbsp;&nbsp; 参数为双List时，默认前x轴后y轴<br>可以填入多组x,y轴执行绘图：<br>&nbsp;&nbsp;&nbsp;&nbsp; pyplot.plot(x1,y1,x2,y2,x3,y3,x4,y4)
- pyplot.axis([Xstart,Xend,,Ystart,Yend])&nbsp;&nbsp; x与y轴的区间范围设定
- pyplot.subplot(nrows,ncols,plot_num)&nbsp;&nbsp; 区域分割为 nrows * ncols 个子区域，在plot_num指定区绘制图形。
- matplotlib.rcParams[''] = ''<br>&nbsp;&nbsp;&nbsp;&nbsp; 参数为font-family时为设置字体<br>&nbsp;&nbsp;&nbsp;&nbsp; 参数为font-size为设置大小<br>&nbsp;&nbsp;&nbsp;&nbsp; 参数为font-style为设置风格<br>&nbsp;&nbsp;&nbsp;&nbsp; 此处为全局设置，单个设置可以在各个单独属性中设置。



## **pyplot 几个需要注意的参数**
- pyplot的format_string格式：<br>&nbsp;&nbsp;&nbsp;&nbsp; '-' : 实线<br>&nbsp;&nbsp;&nbsp;&nbsp; ':' : 虚线<br>&nbsp;&nbsp;&nbsp;&nbsp; '-.' : 点划线<br>&nbsp;&nbsp;&nbsp;&nbsp; '--' : 波折线
- x，ylabel设置中的参数：<br>&nbsp;&nbsp;&nbsp;&nbsp; fontproperties：字体<br>&nbsp;&nbsp;&nbsp;&nbsp; fontsize：大小<br>&nbsp;&nbsp;&nbsp;&nbsp; 此处即为单独修改格式方式