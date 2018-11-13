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
1 2 3 4 5 6 7 8 9 10 11 12
完整代码
该代码实现自《TensorFlow：实战Google深度学习框架》

整个代码的结构如下

lstm_model() 类用于创建 LSTM 网络并返回一些结果
LstmCell() 函数用于创建单层 LSTM 结构，防止 LSTM 参数名称一样
generate_data() 函数用于创建数据集
由于原书中的代码是基于 1.0，而我用的是 1.5，所以出现了很多错误，我将所遇到的错误的解决方法都记录在了文末
--------------------- 


错误总结
1. 没有 unpack
出现如下错误

AttributeError: module 'tensorflow' has no attribute 'unpack'
1
原因是 tf.unpack 改为了 tf.unstack

# 原代码
x_ = tf.unpack(x, axis=1)

# 修改为
x_ = tf.unstack(x, axis=1)
1
2
3
4
5
2. 没有 rnn_cell
出现如下错误

AttributeError: module 'tensorflow.python.ops.nn' has no attribute 'rnn_cell'
1
原因是 tf.nn.rnn_cell 改为了 tf.contrib.rnn

# 原代码
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

# 修改为
lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)
1
2
3
4
5
6
7
3. rnn 不可调用
出现如下错误

TypeError: 'module' object is not callable
1
原因是 tf.nn.rnn 现在改为了几个方法

tf.contrib.rnn.static_rnn
tf.contrib.rnn.static_state_saving_rnn
tf.contrib.rnn.static_bidirectional_rnn
tf.contrib.rnn.stack_bidirectional_dynamic_rnn
1
2
3
4
而我们需要的是 tf.nn.dynamic_rnn() 方法

# 原代码
output, _ = tf.nn.rnn(cell, X, dtype=tf.float32)

# 修改为
output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
1
2
3
4
5
4. 不能调用 Estimator.fit
出现如下警告

WARNING:tensorflow:From train.py:71: calling BaseEstimator.fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with y is deprecated and will be removed after 2016-12-01.
1
该警告下面给出了解决方法

Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
1
2
3
4
5
6
按照给出的方法修改代码

# 原代码
regressor = learn.Estimator(model_fn=lstm_model)

# 修改为
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

regressor = SKCompat(learn.Estimator(model_fn=lstm_model))
1
2
3
4
5
6
7
5. 临时文件夹
出现如下警告

WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmp01x9hws6
1
原因是现在的 Estimator 需要提供 model_dir

# 原代码
regressor = SKCompat(learn.Estimator(model_fn=lstm_model))

# 修改为
regressor = SKCompat(
    learn.Estimator(model_fn=lstm_model, model_dir='model/'))
1
2
3
4
5
6
6. 尺寸必须一致
出现如下错误

ValueError: Dimensions must be equal, but are 60 and 40 
for 'rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1' 
(op: 'MatMul') with input shapes: [?,60], [40,120].
1
2
3
原因我不太清楚，可能是因为 TensorFlow 的调整导致生成的数据在形状上与老版本不一致，也可能是因为使用 lstm_cell*NUM_LAYERS 的方法创建深层循环网络模型导致每层 LSTM 的 tensor 名称都一样

只能在网上搜了其他的类似的博客后照着修改了代码，下面给出了修改的关键地方，详细的部分在完整代码中

# LSTM结构单元
def LstmCell():
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
    return lstm_cell

def lstm_model(X, y):
    # 使用多层LSTM，不能用lstm_cell*NUM_LAYERS的方法，会导致LSTM的tensor名字都一样
    cell = tf.contrib.rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    # 将多层LSTM结构连接成RNN网络并计算前向传播结果
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    ......
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
7. Legend 不支持
出现如下错误

UserWarning: Legend does not support [<matplotlib.lines.Line2D object at 0x7feb52d58c18>] instances.
A proxy artist may be used instead.
1
2
原因是因为需要在调用 plt.plot 时参数解包

# 原代码
plot_predicted = plt.plot(predicted, label='predicted')
plot_test = plt.plot(test_y, label='real_sin')

# 修改为（加逗号）
plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
1
2
3
4
5
6
7
8. 使用 plt.show() 不显示图片
在代码中使用 plt.show() 运行之后没有图片显示，原因是原代码中使用了 mpl.use(‘Agg’)，而 Agg 是不会画图的，所以直接把这一行删掉

9. get_global_step 不建议使用
出现如下警告

WARNING:tensorflow:From train.py:60: get_global_step 
(from tensorflow.contrib.framework.python.ops.variables) 
is deprecated and will be removed in a future version.
1
2
3
警告下面给出了解决方法

Instructions for updating:
Please switch to tf.train.get_global_step
1
2
按照解决方法修改代码

# 原代码
train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1)

# 修改为
train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.train.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1)
--------------------- 
