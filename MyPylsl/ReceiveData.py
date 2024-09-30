"""Example program to show how to read a multi-channel time series from LSL."""
from pylsl import StreamInlet, resolve_stream
import numpy as np


def getmain():
    data = np.zeros([1200, 8])
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    i = 0

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        # print(timestamp, sample)
        data[i] = timestamp
        i = i + 1
        if i == 1200:
            return data


# if __name__ == '__main__':
#     main()
