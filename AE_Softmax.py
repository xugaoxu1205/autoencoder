import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from library.Autoencoder import Autoencoder
from library.utils import fold, next_batch, read_data
import numpy as np

# hyper parameters
# 训练集的样本数量
n_samples = int(378)
# 训练轮数
training_epochs = 100
# 每次读入的样本数
batch_size = 100
display_step = 1

corruption_level = 0.05
sparse_reg = 0.02

# 输入维数
n_inputs = 3752
n_hidden = 600
n_outputs = 2
lr = 0.0001

# 模型定义
ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                 transfer_function=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(learning_rate=lr),
                 ae_para=[corruption_level, sparse_reg])

# 定义softmax的模型
x = tf.placeholder(tf.float32, [None, n_hidden])
W = tf.Variable(tf.zeros([n_hidden, n_outputs]))
b = tf.Variable(tf.zeros([n_outputs]))
# 矩阵乘
y = tf.matmul(x, W) + b

# 定义损失函数以及优化器
y_ = tf.placeholder(tf.float32, [None, n_outputs])
# 交叉熵的损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# Adam优化器
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

## 微调：将ae和分类器连起来需要一个微调的过程，使得每层的预处理参数得到改善
x_ae = ae.transform()
y_ft = tf.matmul(x_ae, W) + b

cross_entropy_ft = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_ft, labels=y_))
train_step_ft = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_ft)

# 准确率等信息
correct_prediction_ft = tf.equal(tf.argmax(y_ft, 1), tf.argmax(y_, 1))
accuracy_ft = tf.reduce_mean(tf.cast(correct_prediction_ft, tf.float32))

if __name__ == '__main__':
    file_path = './data/data.mat'
    # 读出了‘data’这个cell下的数据，维度为42*3752
    data, labels = read_data(file_path)
    # data:10*42*3752 ,labels:10*42*1
    # print(data.shape,labels.shape)
    # 将labels转为独热码
    labels = tf.cast(labels, tf.int32)
    labels = tf.one_hot(labels, 2)
    labels = tf.Session().run(labels)

    # 十折交叉验证开始
    for i_fold in range(10):
        data_train, data_test, labels_train, labels_test = fold(data, labels, i_fold)
        ## 开始训练
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # 遍历所有的batch
            for i in range(total_batch):
                batch_xs, _ = next_batch(data_train, labels_train, batch_size)
                # 训练开始
                temp = ae.partial_fit()
                cost, opt = sess.run(temp, feed_dict={ae.x: batch_xs, ae.keep_prob: ae.in_keep_prob})

                # 平均损失
                avg_cost += cost / n_samples * batch_size

            # 输出日志
            if epoch % display_step == 0:
                print("Fold:", i_fold, "Epoch:", '%d,' % (epoch + 1),
                      "Cost:", "{:.9f}".format(avg_cost))
        # print("Total cost: " + str(ae_test_cost))

        print("######################Finish the autoencoder training######################")

        # Training the softmax layer
        for _ in range(1000):
            batch_xs, batch_ys = next_batch(data_train, labels_train, batch_size)
            # 类型转化
            batch_ys = np.array(batch_ys)
            x_ae = sess.run(ae.transform(), feed_dict={ae.x: batch_xs, ae.keep_prob: ae.in_keep_prob})
            # print(x_ae.shape,batch_ys.shape)
            sess.run(train_step, feed_dict={x: x_ae, y_: batch_ys})

        print("######################Finish the softmax output layer training######################")
        print("Test accuracy before fine-tune")
        print(sess.run(accuracy_ft, feed_dict={ae.x: data_test, y_: labels_test,
                                               ae.keep_prob: 1.0}))

        # Training of fine tune
        for _ in range(1000):
            batch_xs, batch_ys = next_batch(data_train, labels_train, batch_size)
            sess.run(train_step_ft, feed_dict={ae.x: batch_xs, y_: batch_ys, ae.keep_prob: ae.in_keep_prob})
        print("************************Finish the fine tuning******************************")
        # Test trained model

        print(sess.run(accuracy_ft, feed_dict={ae.x: data_test, y_: labels_test, ae.keep_prob: 1.0}))
