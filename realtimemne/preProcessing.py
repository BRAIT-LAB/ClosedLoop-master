import mne
import numpy as np
from CnnP300 import train


def processing(data, num, mark, ts):
    # ts1 = [int(i)*400 for i in ts]
    event = np.zeros([3, num * 12])
    # 采样率400，计算mark数据点
    event[0] = ts * 400
    event[2] = mark
    event = event.T
    event = event.astype(int)
    event_dict = {
        "h1": 6,
        "h2": 9,
        "h3": 3,
        "h4": 12,
        "h5": 1,
        "h6": 8,
        "l1": 4,
        "l2": 10,
        "l3": 2,
        "l4": 7,
        "l5": 5,
        "l6": 11,
    }

    # 预处理
    data.resample(400)
    data.filter(1, 30)
    # data.notch_filter(50)
    # data.plot(duration=8, n_channels=10)
    # 然后，叠加评价，直接平分num份求平均，不需要mark
    # 对mne的数据不会叠加平均，所以转化为numpy进行叠加平均，刚好还能传入深度学习接口
    epochs = mne.Epochs(data, event, tmin=-0.1, tmax=0.5, event_id=event_dict, preload=True)
    # evoked = epochs["h3"].average()
    # evoked.plot()
    hall = np.array([epochs["h1"].average().data, epochs["h2"].average().data, epochs["h3"].average().data,
                     epochs["h4"].average().data,
                     epochs["h5"].average().data, epochs["h6"].average().data]).astype(np.float32)
    lall = np.array([epochs["l1"].average().data, epochs["l2"].average().data, epochs["l3"].average().data,
                     epochs["l4"].average().data,
                     epochs["l5"].average().data, epochs["l6"].average().data]).astype(np.float32)
    getstr, location = train.predictA(hall, lall)
    # print(evoked.data.shape)
    # data = data.get_data()
    # length = data.shape[1] - (data.shape[1] % num)
    #
    # data_list = np.split(data[:, :length], num, axis=1)
    #
    # # 循环将列表内元素叠加平均
    # erpData = sum(data_list)/num
    # print(erpData.shape)

    return getstr, location