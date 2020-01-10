import csv
import numpy as np
import torch


def scaler(data: list) -> np.array:
    data_min = np.nanmin(data, axis=0)
    data_max = np.nanmax(data, axis=0)
    data = (np.array(data) - data_min) / (data_max - data_min)
    return data


def augum(path: str, rows_numers: list, scaling=True):

    data = []

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            line = []
            for x in rows_numers:
                line.append(float(row[x]))
            data.append(line)  # list of selected rows
    return list(zip(*data))


if __name__ == '__main__':
    # 9 rows
    p = '/mnt/hit4/hit4user/PycharmProjects/my_pytorch_lstm/YNDX_191211_191223.csv'
    data = augum(p, [7], scaling=True)
    data = list(range(1500))
    # train_data = data[0][:11000]  # 11000 records
    # test_data = data[0][11000:]



    smoothing_window_size = 1000
    dl = []
    for di in range(0, len(data), smoothing_window_size):
        window = data[di:di + smoothing_window_size]
        window = scaler(window)
        # print(window[0], window[-1])
        dl.append(window)  # last line will be shorter

    data = np.concatenate(dl)
    print(data)
    # print(a)
    # BATCHES
    batch_num = 200
    batches = []
    # 1) strict windows
    # for di in range(0, len(data), batch_num):
    #     window = train_data[di:di + batch_num]
    #     if len(window) < 1000:  # last short window
    #         window = list(window) + [0.] * (batch_num - len(window))
    #     batches.append(window)
    # 2) slide window with stride 1
    for di in range(len(data)-batch_num + 1):
        window = data[di:di + batch_num]
        batches.append(window)

    # data_res = np.transpose(np.array(batches))
    data_res = np.array(batches)
    print(data_res.shape)
    torch.save(data_res, open('traindata_ya.pt', 'wb'))

    # from matplotlib import pyplot as plt
    # plt.plot(range(len(train_data)), a)
    # plt.show()


    # a1 = scaler(d[1])
    # print(a1)
    # print(d)
# data = scale10(data)
# print(np.mean(data), max(data), min(data))

#
# # print(data_min, data_max)
# scale = np.nanstd(data, axis=0)
# data /= scale
# print(np.nanstd(data, axis=0))
# mean = np.nanmean(data, axis=0)
# data -= mean
# print(np.nanmean(data, axis=0))
# print(np.nanstd(data, axis=0))
# #
# # print(data)