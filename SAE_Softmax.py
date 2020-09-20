import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from library.Autoencoder import Autoencoder
from library.utils import fold, next_batch, read_data

# 超参数
n_samples = int(378)
training_epochs = 100
batch_size = 100
display_step = 1
corruption_level = 0.05
sparse_reg = 0.02
n_inputs = 3752
n_hidden = 600
n_hidden2 = 100
n_outputs = 2
lr = 0.001

# 自编码器定义
ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                 transfer_function=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(learning_rate=lr),
                 ae_para=[corruption_level, sparse_reg])
ae_2nd = Autoencoder(n_layers=[n_hidden, n_hidden2],
                     transfer_function=tf.nn.relu,
                     optimizer=tf.train.AdamOptimizer(learning_rate=lr),
                     ae_para=[corruption_level, sparse_reg])

# 定义softmax的模型
x = tf.placeholder(tf.float32, [None, n_hidden2])
W = tf.Variable(tf.zeros([n_hidden2, n_outputs]))
b = tf.Variable(tf.zeros([n_outputs]))
y = tf.matmul(x, W) + b

# 定义损失函数以及优化器
y_ = tf.placeholder(tf.float32, [None, n_outputs])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

## ## 微调：将ae和分类器连起来需要一个微调的过程，使得每层的预处理参数得到改善
x_ft = tf.placeholder(tf.float32, [None, n_inputs])
h = x_ft

# 经过两个编码器
for layer in range(len(ae.n_layers) - 1):
    # h = tf.nn.dropout(h, ae.in_keep_prob)
    h = ae.transfer(
        tf.add(tf.matmul(h, ae.weights['encode'][layer]['w']), ae.weights['encode'][layer]['b']))
for layer in range(len(ae_2nd.n_layers) - 1):
    # h = tf.nn.dropout(h, ae_2nd.in_keep_prob)
    h = ae_2nd.transfer(
        tf.add(tf.matmul(h, ae_2nd.weights['encode'][layer]['w']), ae_2nd.weights['encode'][layer]['b']))

y_ft = tf.matmul(h, W) + b
cross_entropy_ft = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_ft, labels=y_))

train_step_ft = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_ft)
correct_prediction = tf.equal(tf.argmax(y_ft, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
        ## 初始化参数
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # 遍历batch
            for i in range(total_batch):
                batch_xs, _ = next_batch(data_train, labels_train, batch_size)

                # 训练
                temp = ae.partial_fit()
                cost, opt = sess.run(temp, feed_dict={ae.x: batch_xs, ae.keep_prob: ae.in_keep_prob})

                # 平均损失
                avg_cost += cost / n_samples * batch_size

            # 输出日志
            if epoch % display_step == 0:
                print("Fold:", i_fold, "Epoch:", '%d,' % (epoch + 1),
                      "Cost:", "{:.9f}".format(avg_cost))
        ae_test_cost = sess.run(ae.calc_total_cost(), feed_dict={ae.x: data_test, ae.keep_prob: 1.0})
        # print("Total cost: " + str(ae_test_cost))

        print("##########################First AE training finished##########################")

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = next_batch(data_train, labels_train, batch_size)

                # 第一个ae的隐层作为输入
                h_ae1_out = sess.run(ae.transform(), feed_dict={ae.x: batch_xs, ae.keep_prob: ae.in_keep_prob})
                temp = ae_2nd.partial_fit()
                cost, opt = sess.run(temp, feed_dict={ae_2nd.x: h_ae1_out, ae_2nd.keep_prob: ae_2nd.in_keep_prob})

                avg_cost += cost / n_samples * batch_size

            if epoch % display_step == 0:
                print("Fold:", i_fold, "Epoch:", '%d,' % (epoch + 1),
                      "Cost:", "{:.9f}".format(avg_cost))

        print("##########################Second AE training finished##########################")

        # softmax层，迭代1000次，没有输出这层的日志
        for _ in range(1000):
            batch_xs, batch_ys = next_batch(data_train, labels_train, batch_size)
            h_ae1_out = sess.run(ae.transform(), feed_dict={ae.x: batch_xs, ae.keep_prob: 1.0})
            h_ae2_out = sess.run(ae_2nd.transform(), feed_dict={ae_2nd.x: h_ae1_out, ae_2nd.keep_prob: 1.0})
            sess.run(train_step, feed_dict={x: h_ae2_out, y_: batch_ys})
        print("##########################Finish the softmax output layer training##########################")

        print("Test accuracy before fine-tune")
        print(sess.run(accuracy, feed_dict={x_ft: data_test, y_: labels_test,
                                            ae.keep_prob: 1.0, ae_2nd.keep_prob: 1.0}))

        # 微调
        for _ in range(1000):
            batch_xs, batch_ys = next_batch(data_train, labels_train, batch_size)
            sess.run(train_step_ft, feed_dict={x_ft: batch_xs, y_: batch_ys,
                                               ae.keep_prob: 1.0, ae_2nd.keep_prob: 1.0})

        print("##########################Finish the fine tuning##########################")
        # 测试
        print(sess.run(accuracy, feed_dict={x_ft: data_test, y_: labels_test,
                                            ae.keep_prob: 1.0, ae_2nd.keep_prob: 1.0}))
