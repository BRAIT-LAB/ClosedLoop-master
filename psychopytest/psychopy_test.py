from psychopy import visual, core, event
import get_
from pylsl import StreamInfo, StreamOutlet
from pylsl import StreamInlet, resolve_stream
from RL import work_DQN

info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
outlet = StreamOutlet(info)
info2 = StreamInfo('MyMarkerStream1', 'Markersall', 1, 0, 'float32', 'myuidw43537')
outlet2 = StreamOutlet(info2)

# 接收预测P300准确率
streams = resolve_stream('type', 'myresult')
inlet = StreamInlet(streams[0])

# 961.541图片大小
# 设置2不动，无论窗口怎么变，刚好可以填满窗口
allsize = 2
win = visual.Window(size=(1442, 811))

b0 = r"../background/other/0.png"
b1 = r"../background/other/p1.png"
b2 = r"../background/other/p2.png"
b3 = r"../background/other/p3.png"
b4 = r"../background/other/p4.png"
b5 = r"../background/other/p5.png"
b6 = r"../background/other/p6.png"
b7 = r"../background/other/p7.png"
b8 = r"../background/other/p8.png"
b9 = r"../background/other/p9.png"
b10 = r"../background/other/p10.png"
b11 = r"../background/other/p11.png"
b12 = r"../background/other/p12.png"

an1 = "../background/A.png"
an2 = "../background/N.png"
hui1 = "../background/H.png"
hui2 = "../background/U.png"
hui3 = "../background/I.png"
da1 = "../background/D.png"
da2 = "../background/A.png"
xue1 = "../background/X.png"
xue2 = "../background/U.png"
xue3 = "../background/E.png"

# image_ = [b1, b9, b3, b12, b6, b4, b7, b10, b2, b8, b11, b5]
image_ = [b6, b9, b3, b12, b1, b8, b4, b10, b2, b7, b5, b11]
clue = [an1, an2, hui1, hui2, hui3, da1, da2, xue1, xue2, xue3]

image = visual.ImageStim(win, b0, size=allsize)

image2 = [visual.ImageStim(win, image_[0], size=allsize), visual.ImageStim(win, image_[1], size=allsize), visual.ImageStim(win, image_[2], size=allsize),
          visual.ImageStim(win, image_[3], size=allsize), visual.ImageStim(win, image_[4], size=allsize), visual.ImageStim(win, image_[5], size=allsize),
          visual.ImageStim(win, image_[6], size=allsize), visual.ImageStim(win, image_[7], size=allsize), visual.ImageStim(win, image_[8], size=allsize),
          visual.ImageStim(win, image_[9], size=allsize), visual.ImageStim(win, image_[10], size=allsize), visual.ImageStim(win, image_[11], size=allsize)]
clue_show = [visual.ImageStim(win, clue[0], size=allsize), visual.ImageStim(win, clue[1], size=allsize), visual.ImageStim(win, clue[2], size=allsize),
             visual.ImageStim(win, clue[3], size=allsize), visual.ImageStim(win, clue[4], size=allsize), visual.ImageStim(win, clue[5], size=allsize),
             visual.ImageStim(win, clue[6], size=allsize), visual.ImageStim(win, clue[7], size=allsize), visual.ImageStim(win, clue[8], size=allsize),
             visual.ImageStim(win, clue[9], size=allsize)]

text = visual.TextStim(win, "开始")
text.draw()
win.flip()
core.wait(3)

# 闪光时间
flashTime = 0.15
# 闪烁次数
flashNum = 10
# 开局准确率
old_attention = 0.3
realword = "ANHUIDAXUE"
for wai in range(10):
    outlet.push_sample(['begin'])
    # 中间休息30秒
    if wai == 5:
        text = visual.TextStim(win, "休息30秒")
        text.draw()
        win.flip()
        core.wait(30)

    # 展示提示字符
    clue_show[wai].draw()
    win.flip()
    core.wait(2)

    for i in range(16):
        for j in range(12):
            # 全黑
            image.draw()
            win.flip()
            core.wait(flashTime)

            # 高亮目标行、列
            outlet2.push_sample([j + 1])
            image2[j].draw()
            win.flip()
            core.wait(flashTime)
        if i == flashNum - 1:
            print(i, '结束了')
            break

    outlet.push_sample(['end'])
    text = visual.TextStim(win, "开始计算数据")
    # 传送下轮闪烁轮数
    text.draw()
    win.flip()

    my_rnn, _ = inlet.pull_sample()

    text = visual.TextStim(win, "得分" + my_rnn[0][:-1] + "\n"+"预测字母：" + my_rnn[0][-1]+"真实字母："+realword[wai])
    # 调用强化学习
    flashNum, attention = work_DQN.RL_num(flashNum, old_attention, my_rnn[0][:-1])
    # 传送下轮闪烁轮数
    text.draw()
    win.flip()
    # 再将得到的闪烁次数传递给后台函数
    outlet.push_sample([str(flashNum)])
    print("下轮闪烁次数",flashNum)
    core.wait(5)

win.close()
