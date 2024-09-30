import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat

#
my_need = [9, 11, 13, 22, 23, 24, 30, 32, 34, 36, 38, 41, 42, 47, 49, 51, 53, 55, 61, 62, 63]
class Trainset(Dataset):
    def __init__(self, subject_name):
        if subject_name == 'A':
            raw_data = loadmat('dataset/sub_a.mat')
        else:
            raw_data = loadmat('dataset/sub_b.mat')

        # singal是12 240 64 85
        # 12行列闪烁 64电极 85个字母 240是时间维度？

        signals = raw_data['responses'][:, :, my_need, :]
        print(signals.shape)
        # label是0和1，亮的行是1，总共85个label
        # shape(12,85)
        label = raw_data['is_stimulate']
        # for i in range(12):
        #     print(sum(label[i]))
        data = []
        target = []

        # 12次闪烁
        for i in range(12):
            for j in range(85):
                if label[i, j] == 1:
                    # 增加正样本的数量，正样本1例变5例
                    data.append(signals[i, :, :, j].reshape(-1, 21))
                    target.append(label[i, j])
                    data.append(signals[i, :, :, j].reshape(-1, 21))
                    target.append(label[i, j])
                    data.append(signals[i, :, :, j].reshape(-1, 21))
                    target.append(label[i, j])
                    data.append(signals[i, :, :, j].reshape(-1, 21))
                    target.append(label[i, j])
                # reshape为240，64
                data.append(signals[i, :, :, j].reshape(-1, 21))
                target.append(label[i, j])
        # 此处一个data意味着什么？
        self.data = np.array(data)
        self.target = np.array(target)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :].astype(np.float32).T, self.target[index].astype(np.float32)


class Testset(Dataset):
    def __init__(self, subject_name):
        if subject_name == 'A':
            raw_data = loadmat('dataset/sub_a_test.mat')
        else:
            raw_data = loadmat('dataset/sub_b_test.mat')
        # shape (12, 240, 64, 100)
        self.signals = raw_data['responses'][:, :, my_need, :]

    def __len__(self):
        return self.signals.shape[-1]

    def __getitem__(self, index):
        col = self.signals[:6, :, :, index].astype(np.float32).transpose([0, 2, 1])
        row = self.signals[6:, :, :, index].astype(np.float32).transpose([0, 2, 1])
        return col, row

def lcy():
    print(1)