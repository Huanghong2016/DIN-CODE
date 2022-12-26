## RNN 的基本原理+pytorch代码



[TOC]

### 1.RNN模型的结构

传统的神经网络结构如下，由输入层，隐藏层和输出层组成

![image-20220912121608317](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220912121608317.png)

而RNN跟传统神经网络的最大区别在于每次都会将前一次的隐藏层的结果带到下一次隐藏层的计算中，如图所示：

![RNN动图](C:\Users\Administrator\Pictures\Saved Pictures\RNN动图.gif)

如果我们将这个循环展开：

![img](https://img-blog.csdnimg.cn/img_convert/f6cdd1b5ff8c6ca0cad2f6afcea8f635.png)



上图是一个RNN模型的结构，RNN是循环神经网络，所指的x~t~和h~t~

就是t时刻的输入，和t时刻对应的隐藏层，千万不要理解成隐藏层有t个隐藏单元。再说一次t是指时间维度。

![image-20220912165229769](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220912165229769.png)

举例：家里电脑坏了，我要拿去修，修好要200大洋。

此时，x~t-1~为电脑，h~t-1~=某件物品

​			x~t~为修，h~t~可以根据x~t~和h~t-1~学习到要去修电脑

​			x~t+1~为200大洋，h~t+1~可以学习到修xxx要200大洋

为啥h~t+1~不知道要修啥呢？200大洋和电脑之间距离太远忘记了（长依赖问题）。

### 2.模型输入（inputs）

RNN模型主要应用于时序型的数据，常见的应用类型为

a.如自然语言：你恰饭了没啊，x~1~为你，x~2~为恰...以此类推

b.股票价格，每日天气情况等

对于该类数据，传统模型无法表达前一个状态与后一个状态之间的连续性，故我们一般采用序列型的模型。



### 3.隐藏层（hidden）

弄清楚了模型输入后，继续了解隐藏层的内容，h~t~为t时刻输入x对应的隐藏单元
$$
h_t=tanh(Ux_t+Wh_{t-1}+b)
$$
其中

h~t~为t时刻时输入对应的隐藏单元的值；

U是输入层到隐藏层的权重矩阵；

W就是上一次隐藏层的值h~t-1~作为本次输入的权值矩阵。；

b为偏置量



### 4.输出层（output）

得到隐藏层的内容后即可计算输出结果
$$
y_t=softmax(Vh_t+c)
$$
其中

V是隐藏层到输出层的权重矩阵。

c为偏置量



注：在计算时，每一步使用的参数U、W、V、b、c都是一样的，也就是说每个步骤的参数都是共享的；



### 5.反向传播

其实反向传播就是对U、W、V、b、c求偏导，调整他们是的损失变小。

设t时刻，损失函数为
$$
L_t=\frac12(y_t-y^‘_t)^2
$$


则，损失函数之和为
$$
L = \sum_{t=0}^T{L_t}
$$
W在每一个时刻都出现了，所以W在t时刻的梯度=时刻t所有损失函数对对所有时刻的w的梯度之和：
$$
\frac{\partial{L}}{\partial{W}} = \sum_{t=0}^T{\frac{\partial{L_t}}{\partial{W}} }=\sum_{t=0}^T{\sum_{s=0}^T{\frac{\partial{L_t}}{\partial{W_s}} }}
$$
最后更新参数
$$
W=W-\alpha\frac{\partial{L}}{\partial{W}}
$$
接下来举个例子，t=2时刻U、V、W对于损失函数L~2~的偏导：
$$
\frac{\partial{L_2}}{\partial{U}} = {\frac{\partial{L_2}}{\partial{y_2}} }{\frac{\partial{y_2}}{\partial{h_2}} }{\frac{\partial{h_2}}{\partial{U_2}} }+{\frac{\partial{L_2}}{\partial{y_2}} }{\frac{\partial{y_2}}{\partial{h_2}} }{\frac{\partial{h_2}}{\partial{h_1}} }{\frac{\partial{h_1}}{\partial{U_1}} }
$$

$$
\frac{\partial{L_2}}{\partial{V}} = {\frac{\partial{L_2}}{\partial{y_2}} }{\frac{\partial{y_2}}{\partial{V}} }
$$

$$
\frac{\partial{L_2}}{\partial{W}} = {\frac{\partial{L_2}}{\partial{y_2}} }{\frac{\partial{y_2}}{\partial{h_2}} }{\frac{\partial{h_2}}{\partial{W}} }+{\frac{\partial{L_2}}{\partial{y_2}} }{\frac{\partial{y_2}}{\partial{h_2}} }{\frac{\partial{h_2}}{\partial{h_1}} }{\frac{\partial{h_1}}{\partial{W}} }
$$

以W，归纳总结后为
$$
\frac{\partial{L_t}}{\partial{W}} = \sum_{t=0}^t{\frac{\partial{L_t}}{\partial{y_t}} }{\frac{\partial{y_t}}{\partial{h_t}} }(\prod_{j=k+1}^{t}{tanh'W}){\frac{\partial{h_j}}{\partial{W}} }
$$
tanh函数及其导数图像

![image-20220912141430809](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220912141430809.png)

故，当t很大时，时间比较前的输入对当前损失的影响会很小，也就是梯度消失。



### 6.RNN的缺陷：长依赖问题

RNN的优势为可以利用过去的数据来推测当前数据的理解方式，但是由于RNN的参数是共享的，每一时刻都会由前面所有的时刻共同决定，这是一个相加的过程，这样的话就有个问题，当距离过长了，计算前面的导数时，最前面的导数就会消失或者爆炸，当但是当前时刻的整体梯度并不会消失，因为t时刻隐藏单元的值是1、2.....t-1,时刻的值传过来的。且由于权值共享，所以整体的梯度还是会更新，通常在RNN中说的梯度消失是指后面的信息用不到前面的信息了。

所以当相关的数据离推测的数据越远时，RNN所能学习到的信息则越少。

例如：I live in Beijing.  .......  .I can speak Chinese.

Beijing和Chinese是有着密切的关系的，但是由于中间存在着大量的句子，导致识别到Chinese无法和前面的Beijing产生联系。



### 7.pytorch调用RNN
​        ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200206114608800.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VjaG9vb1poYW5n,size_16,color_FFFFFF,t_70)

```python
#pytorch调用RNN代码
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        #隐藏层特征数量
        self.hidden_dim=hidden_dim
        '''
        input_size    – 输入x中预期特征的数量
		hidden_size   – 隐藏状态h的特征数量
		num_layers    - 循环层数。例如，设置 num_layers=2意味着将两个RNN 堆叠在一起形成一个堆叠的RNN，第二个RNN接收						    第一个RNN的输出并计算最终结果。默认值：1
		nonlinearity  — 隐藏层函数，可以是“tanh”或“relu”。默认值：'tanh'
		bias 		  - 如果为 False，则该层不使用偏差权重。默认值：真
		batch_first   - 输入特征是不是批量输入。默认值False
		dropout 	  - 是否要引入Dropout层，dropout概率等于dropout。默认值：0
		bidirectional —如果为真，则成为双向 RNN。默认值：假
		'''
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        #全连接层
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        #批量输入大小
        batch_size = x.size(0)
        #批量输入input，隐藏层参数
        r_out, hidden = self.rnn(x, hidden)
        #维度转化
        r_out = r_out.view(-1, self.hidden_dim)  
        #全连接
        output = self.fc(r_out)
        return output, hidden
```




参考

1.[pytorch RNN代码分析](https://blog.csdn.net/qq_35272180/article/details/115765269)

2.[如何从RNN起步，一步一步通俗理解LSTM](https://blog.csdn.net/v_JULY_v/article/details/89894058)

3.[通俗易懂的RNN](https://blog.csdn.net/qq_39439006/article/details/121554808)

4.[Pytorch菜鸟入门（5）——RNN入门【代码】](https://blog.csdn.net/EchoooZhang/article/details/104193945)

5.[【循环神经网络】5分钟搞懂RNN，3D动画深入浅出](https://www.bilibili.com/video/BV1z5411f7Bm?spm_id_from=333.337.search-card.all.click&vd_source=0dc9529a3d38331f20d92486fa5c5022)
