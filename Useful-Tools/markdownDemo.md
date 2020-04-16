# Markdown Demo

> 用于描述记录一些markdown的基本常用操作语法。

---

## Basic 基础操作

### Font 字体格式

- 加粗：**Sample** or __Sample__
- 斜体：*Sample* or _Sample_
- 强调：`Sample`
- 删除：~~Sample~~
- 可多种结合使用：_**~~~Sample~~~**_

### Image & URL 图片与超链接

- URL: [URL_NAME](# "URL_Description")
- IMG：![IMG_NAME][IMG_URL]

[IMG_URL]:IMG_SOURCE

### List & Paragraph 列表与段落

- Sub title level 1-1
  - Sub title level 2-1
    > Paragraph
    >> Sub Paragraph level 1
    >>> Sub Paragraph level 2
    >>> - Build list in level 2 Para topic 1
    >>> - Build list in level 2 Para topic 2
    >>> ---
    >>> - Change <br> The lines
    >>> ---
    >> *Back to the level 1 sub paragraph*
    >> 
    > __End__
  - Sub title level 2-2
- Sub title level 1-2

### Tables 表格

| 左对齐         |       居中       |             右对齐 |
| :------------- | :--------------: | -----------------: |
| _靠左详细内容_ | **居中详细内容** | `~~靠右详细内容~~` |

### Code blocks 代码区块

```Python
class RNN(torch.nn.Module) :
    def __init__(self) :
        super(RNN, self).__init__()
        # use long short time memory rnn
        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1, # define how many layers in one cell
            batch_first=True # put the batch parameter at first place
        )
        self.out = torch.nn.Linear(64, 10)

    def forward(self, x) :
        rnn_out, (hid_n, hid_c) = self.rnn(x, None)
        output = self.out(rnn_out[:, -1, :]) # (batch, timestep, input)
        return output
```

---

## Latex 公式编写

- 行内公式示例：$\sum_{i=1}^K \frac{W_i \times l_i + bias}{P_i \times l_i}$

- 居中矩阵示例：
$$
\begin{aligned} 
\left [
\begin{matrix}
    a_11 & a_12 & a_13 \\
    a_21 & a_22 & a_23 \\
    a_31 & a_32 & a_33 
\end{matrix}
\right ]
= a_{11}
\left [
\begin{matrix}
    a_22&a_23 \\
    a_32&a_33 
\end{matrix}
\right ]
+ a_{12}
\left [
\begin{matrix}
    a_21 & a_23 \\
    a_31 & a_33
\end{matrix}
\right ]
+ a_{13}
\left [
\begin{matrix}
    a_21 & a_23 \\
    a_31 & a_32
\end{matrix}
\right ]
\end{aligned}
$$

---

## Html 嵌入Html代码

### 抽屉示例

<Details>

<summary>Click to expend</summary>

- More info 1

- More info 2
  
- More info 3
  
- More info 4
  
</Details>

> Html标签在markdown中基本都得到支持，但需要注意是否需要添加如 [html] 标注，或当前markdown在线编译器无法支持Html语法（如微软sharedocs）

---