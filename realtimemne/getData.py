import numpy as np
import mne
import MyPylsl.ReceiveData as lcydata
import matplotlib.pyplot as plt
import preProcessing
from pylsl import StreamInlet, resolve_stream
from mne_lsl.stream import StreamLSL as Stream
from pylsl import StreamInfo, StreamOutlet
import get_score

# 模拟data
from mne_lsl.player import PlayerLSL as Player
player = Player(r"../data/lcy2.vhdr", annotations=False, chunk_size=3, name="EEG")
player.start()

# 给实验范式函数传递准确率
rnn_result = StreamInfo('MyMarkerStream1', 'myresult', 1, 0, 'string', 'myuidw435371')
my_rnn_result = StreamOutlet(rnn_result)

# 用来获取时间的lsl流
streams = resolve_stream('type', 'Markers')
inlet = StreamInlet(streams[0])

# 创建脑电数据的基本信息
n_channels = 21
sampling_freq = 500
info = mne.create_info(n_channels, ch_types="eeg", sfreq=sampling_freq)

# 用来接收数据和mark,接收的是【31，****】维度的数组,stype="EEG",
bufsize = 1024
stream = Stream(bufsize=bufsize, name="EEG").connect()
# 此处的60，指的是缓冲区能存60个数据
stream2 = Stream(bufsize=300, stype="Markersall").connect()
# 首轮闪烁次数
markNum = ['10']
# 真实值
real_word = "ANHUIDAXUE"
for i in range(10):
    # 获得开始和结束的时间戳
    sample, timestamp = inlet.pull_sample()
    sample2, timestamp2 = inlet.pull_sample()

    # 根据时间戳从缓冲区读取数据
    print("时间", timestamp2 - timestamp)
    data, ts = stream.get_data(timestamp2 - timestamp)

    # 一行闪烁markNum次，共12行
    data2, ts2 = stream2.get_data(int(markNum[0]) * 12)
    # print(1111111111111111111111111111111111111)
    # print(data.shape)
    # for i in range(data.shape[1]-1):
    #     print(ts[i+1]-ts[i], data[1,i+1]-data[1,i])
    # print(ts)
    # print("a")
    # print(timestamp,timestamp2)
    # print(ts2)
    # 模拟数据维度不够，所以使用叠加，凑够21导
    # 使用脑电设备模拟器的时候，data替换为np.vstack((data, data, data[1]))
    raw = mne.io.RawArray(data[:21], info)

    get_rnn, location = preProcessing.processing(raw, int(markNum[0]), data2, ts2 - timestamp)

    score = get_score.get(location, real_word[i])

    # 发送准确率
    my_rnn_result.push_sample([str(score) + get_rnn])

    # mne.export.export_raw('data' + str(i) + '.set', raw, overwrite=True)
    markNum, _ = inlet.pull_sample()
    print("现在是第", i, '轮结束', markNum)

# simulated_raw.plot(show_scrollbars=False, show_scalebars=False)
