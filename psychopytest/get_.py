import random
import time


def get_str():
    time.sleep(7)
    return random.randint(5, 7)

    # 这里是需要强化学习计算参数，然后返回下一轮闪烁值
    # 需要调用其他函数，毕竟data在getData那里