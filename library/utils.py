import h5py
import numpy as np


# 这些话实现了读取某折数据中的某一被试的数据
def read_data(file_path):
    final_data = []
    final_labels = []
    dict_data = h5py.File(file_path, 'r')

    data = dict_data['data']
    labels = dict_data['labels']
    # 打印维度发现是转置了故转回来
    data = np.transpose(data)
    labels = np.transpose(labels)
    for i in range(10):
        number_data = data[i][0]
        number_labels = labels[i][0]
        cell_data = dict_data[number_data]
        cell_labels = dict_data[number_labels]
        cell_data = np.transpose(cell_data)
        cell_labels = np.transpose(cell_labels)
        # print(cell_data[:][:])
        final_data.append(cell_data)
        final_labels.append(cell_labels)
    final_data = np.array(final_data)
    final_labels = np.array(final_labels)
    return final_data, final_labels


# 加载训练数据以及测试数据
# 10折交叉验证，因为折都划好了，直接在逻辑上实现就行,i代表将几号元素作为测试集
def fold(data, labels, i):
    # 去除尾特殊元素
    if i == 0:
        data_test = data[0]
        data_train = data[1:]
        labels_test = labels[0]
        labels_train = labels[1:]
    elif i == 9:
        data_test = data[9]
        data_train = data[0:9]
        labels_test = labels[9]
        labels_train = labels[0:9]
    else:
        data_test = data[i]
        data_train = data[0:i]
        data_train = np.append(data_train, data[i + 1:], axis=0)
        labels_test = labels[i]
        labels_train = labels[0:i]
        labels_train = np.append(labels_train, labels[i + 1:], axis=0)
    data_train = data_train.reshape(-1, 3752)
    data_test = data_test.reshape(-1, 3752)
    labels_train = labels_train.reshape(378, 2)
    labels_test = labels_test.reshape(42, 2)

    return data_train, data_test, labels_train, labels_test
    # print('data_train', data_train.shape)
    # print('labels_train',labels_test.shape)


# 随机读取数据，返回标签和读取的数据，train_data训练集特征，train_labels训练集对应的标签，batch_size指读入大小
def next_batch(train_data, train_labels, batch_size):
    # 打乱数据集
    index = [i for i in range(0, 378)]
    np.random.shuffle(index)
    # 建立batch_data与batch_target的空列表
    batch_data = []
    batch_labels = []
    # 向空列表加入训练集及标签
    for i in range(0, batch_size):
        batch_data.append(train_data[index[i]])
        batch_labels.append(train_labels[index[i]])
    return batch_data, batch_labels  # 返回
