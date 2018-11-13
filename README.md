# -TFlearn-sin-
使用TFlearn预测sin曲线
数据生成
因为标准的循环神经网络模型预测的是离散的数值，所以需要将连续的 sin 函数曲线离散化

所谓离散化就是在一个给定的区间 [0,MAX] 内，通过有限个采样点模拟一个连续的曲线，即间隔相同距离取点

采样用的是 numpy.linspace() 函数，它可以创建一个等差序列，常用的参数有三个

start：起始值
stop：终止值，不包含在内
num：数列长度，默认为 50
然后使用一个 generate_data() 函数生成输入和输出，序列的第 i 项和后面的 TIMESTEPS-1 项合在一起作为输入，第 i + TIMESTEPS 项作为输出

TFLearn使用
TFlearn 对训练模型进行了一些封装，使 TensorFlow 更便于使用，如下示范了 TFLearn 的使用方法

from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

learn = tf.contrib.learn

# 建立深层循环网络模型
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir='model/'))

# 调用fit函数训练模型
regressor.fit(train_x, train_y, batch_size=BATCH_SIZE, steps=TRAINGING_STEPS)

# 使用训练好的模型对测试集进行预测
predicted = [[pred] for pred in regressor.predict(test_x)]
1
2
3
4
5
6
7
8
9
10
11
12
完整代码
该代码实现自《TensorFlow：实战Google深度学习框架》

整个代码的结构如下

lstm_model() 类用于创建 LSTM 网络并返回一些结果
LstmCell() 函数用于创建单层 LSTM 结构，防止 LSTM 参数名称一样
generate_data() 函数用于创建数据集
由于原书中的代码是基于 1.0，而我用的是 1.5，所以出现了很多错误，我将所遇到的错误的解决方法都记录在了文末
--------------------- 
作者：widiot8023 
来源：CSDN 
原文：https://blog.csdn.net/white_idiot/article/details/78882856 
版权声明：本文为博主原创文章，转载请附上博文链接！
