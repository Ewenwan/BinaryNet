# Train-time

## Motivations

This subrepository enables the reproduction of the benchmark results reported in the article:  
[BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)

## Requirements

* Python 2.7, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html)
  [我的Theano](https://github.com/Ewenwan/Theano)
* A fast Nvidia GPU (or a large amount of patience)
* Setting your [Theano flags](http://deeplearning.net/software/theano/library/config.html) to use the GPU
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
  
  [Pylearn2 是建立在Theano之上的一个机器学习库](https://github.com/Ewenwan/pylearn2)

* [下载数据集Downloading the datasets](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/datasets) you need

* [数据集工具 PyTables](http://www.pytables.org/usersguide/installation.html) (only for the SVHN dataset)

* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)

## 手写字体数据集测试 MNIST MLP

    python mnist.py
    
This python script trains an MLP on MNIST with BinaryNet.
It should run for about 6 hours on a Titan Black GPU.
The final test error should be around **0.96%**.

    数据集
      60K 28×28的训练集
      10k 28×28的测试集。
      
    Theano设置
      3个4096维的隐含层
      L2-SVM的输出层
      使用Dropout
      ADAM学习法则
      指数衰减的步长
      mini-batch为100的BN
      训练集的最后10k样本作为验证集来early-stopping和模型选择
      大概1000次迭代后模型最好，没有在验证集上重新训练。
      
    Torch7设置
    
      与上面设置的区别：
      没有dropout
      隐含层数目变为2048
      使用shift-based AdaMax和BN
      每十次迭代步长一次右移位


## 10类别图像分类数据集 CIFAR-10 ConvNet

    python cifar10.py
    
This python script trains a ConvNet on CIFAR-10 with BinaryNet.
It should run for about 23 hours on a Titan Black GPU.
The final test error should be around **11.40%**.

    数据集
      50K 32×32的训练集
      10K 32×32的测试集
    Theano设置
      没有任何的数据预处理
      网络结构和Courbariaux 2015的结构一样，除了增加了binarization
      ADAM学习法则
      步长指数损失
      参数初始化来自Glorot & Bengio的工作
      mini-batch为50的BN
      5000个样本作为验证集
      500次迭代后得到最好效果，没有在验证集上重新训练

    Torch7设置
      与上面设置的不同：

      使用shift-based AdaMax和BN（mini-batch大小200）
      每50次迭代，学习率右移一位。


## 街拍门牌号码数据集(SVHN) SVHN ConvNet

    python svhn.py
    
This python script trains a ConvNet on SVHN with BinaryNet.
It should run for about 2 days on a Titan Black GPU.
The final test error should be around **2.80%**.


    数据集
      604K 32×32的训练集
      26K  32×32的测试集

    设置
      基本与cifar10的设置相同，区别如下：

      卷积层只使用一半的单元。
      训练200次迭代就停了，因为太大。

## 总结
    缺点：BNN在训练过程中仍然需要保存实数的参数，这是整个计算的瓶颈。
    个人直观感受：

    BNN虽然需要保存实数的参数，但是实数范围是[-1,1]，所以可以做压缩，
      即使用16bit或者更少的位数来表示浮点数。
    模型尺寸变小，正向传播计算速度变快，意味着可以将正向传播层放到客户端去做了，
      虽然随着网络带宽的增大，给服务器传个图片也么啥。
    将图像的特征学习和哈希码学习可以无缝整合到一起，因为都是二值化。
