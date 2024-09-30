import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from CnnP300.Model import Vanilla, AutoEncoder, InstructedAE, ResCNN, RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Vanilla().to(device)

mse_criterion = nn.MSELoss()
cross_entropy_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9,
                      weight_decay=1e-4)

state = torch.load(r'../CnnP300/model.pth')
model.load_state_dict(state)

model.eval()
print("加载完毕")


def predictA(datah, datal):
    char = [['A', 'B', 'C', 'D', 'E', 'F'],
            ['G', 'H', 'I', 'J', 'K', 'L'],
            ['M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X'],
            ['Y', 'Z', '1', '2', '3', '4'],
            ['5', '6', '7', '8', '9', '_']]


    test_loaderh = DataLoader(dataset=datah, shuffle=False)
    test_loaderl = DataLoader(dataset=datal, shuffle=False)
    with torch.no_grad():
        col_pred_set = []
        row_pred_set = []
        # shape [1, 6, 64, 240]
        # print("     0 - 6")
        # 因为把行和列分开了，所以两个循环索引都是range(6)

        for data in test_loaderh:
            data = data[:, :, :240].to(device)
            output, _ = model(data)
            col_pred_set.append(output.data.cpu().numpy())
        for data in test_loaderl:
            data = data[:, :, :240].to(device)
            output, _ = model(data)
            row_pred_set.append(output.data.cpu().numpy())

        col_pred_set = np.array(col_pred_set).squeeze()
        row_pred_set = np.array(row_pred_set).squeeze()
        # 因为训练的时候就是取1，所以测试也取1
        col_pred = np.argmax(col_pred_set, axis=0)[1]
        row_pred = np.argmax(row_pred_set, axis=0)[1]
    # print("yvcedaode", char[row_pred][col_pred])
    return char[row_pred][col_pred], (row_pred, col_pred)
