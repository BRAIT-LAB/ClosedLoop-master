import numpy as np


def get(location, real_word):
    char = [['A', 'B', 'C', 'D', 'E', 'F'],
            ['G', 'H', 'I', 'J', 'K', 'L'],
            ['M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X'],
            ['Y', 'Z', '1', '2', '3', '4'],
            ['5', '6', '7', '8', '9', '_']]

    real_ = np.where(np.char.find(char, real_word) != -1)
    real_location = [real_[0][0], real_[1][0]]
    # 使用int是为了传输数据方便，接收后又进行了/100
    if real_location == location:
        return 100
    elif location[0] - real_location[0] == 0 or location[1] - real_location[1] == 0:
        return 50
    elif abs(location[0] - real_location[0]) + abs(location[1] - real_location[1]) == 2:
        return 25
    else:
        return 0
