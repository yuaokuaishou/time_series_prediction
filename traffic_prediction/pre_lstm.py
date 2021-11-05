import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data

from sklearn.preprocessing import MinMaxScaler

from data_read import read_data


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

path = "/Users/yuao/Downloads/"
file_name = "traffic_direction.csv"

data_in, data_out = read_data(path, file_name)
data = data_out  # 选择in or out

data_clean = data.drop(["p_date", "p_hourmin", "direction", "avg_speed"],
                       axis=1)  # , "per_0.25", "per_0.5", "per_0.75", "per_0.9"
# data_clean["sum_rate"] = data_clean["sum_rate"] / 10000
data = np.array(data_clean.values.tolist())

# ====================================================
TRAIN_SIZE = 0.8
TRAIN_WINDOW = 24  # 24 = 1天，24 * 7 = 1周，24 * 30 = 1月
PRE_COLUMN = 1  # 0: avg，1：max, 2:min, 3:sum

INPUT_SIZE = len(data[0])
HIDDEN_LAYER_SIZE = 100
OUTPUT_SIZE = 1
EPOCHS = 24  # 24：一天，10：一个月
# ====================================================

train_size = int(len(data) * TRAIN_SIZE)
train = data[:train_size]
test = data[train_size:]

scaler = MinMaxScaler()
train_nor = scaler.fit_transform(train)
test_nor = scaler.fit_transform(test)

train_shape = INPUT_SIZE
train_x = torch.tensor(train_nor[:-TRAIN_WINDOW, :].reshape(-1, train_shape), dtype=torch.float32)
train_y = torch.tensor(train_nor[TRAIN_WINDOW:, PRE_COLUMN].reshape(-1, 1), dtype=torch.float32)
test_x = torch.tensor(test_nor[:-TRAIN_WINDOW, :].reshape(-1, train_shape), dtype=torch.float32)
test_y = torch.tensor(test_nor[TRAIN_WINDOW:, PRE_COLUMN].reshape(-1, 1), dtype=torch.float32)

train_dataset = Data.TensorDataset(train_x, train_y)
test_dataset = Data.TensorDataset(test_x, test_y)

train_loader = Data.DataLoader(dataset=train_dataset)
test_loader = Data.DataLoader(dataset=test_dataset)


class LSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=OUTPUT_SIZE):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)  # lstm层

        self.linear = nn.Linear(hidden_layer_size, output_size)  # 全连接层

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),  # hidden_cell层
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # lstm处理序列数据，并传递到hidden_cell，输出lstm_out
        # 输入数据格式：input(seq_len, batch, input_size)
        # seq_len：每个序列的长度
        # batch_size:设置为1
        # input_size:输入矩阵特征数
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)

        # 全连接层输出predictions
        predictions = self.linear(lstm_out.view(len(input_seq), -1))

        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器

for i in range(EPOCHS):
    for seq, labels in train_loader:
        optimizer.zero_grad()

        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    print(f'epoch:{i:3}loss: {single_loss.item():10.8f}')

torch.save(model, "/Users/yuao/PycharmProjects/traffic_prediction/model/day_out_max.pkl")

model = torch.load("/Users/yuao/PycharmProjects/traffic_prediction/model/day_out_max.pkl")
model.eval()

output_list = np.array([])
for seq, labels in test_loader:
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                         torch.zeros(1, 1, model.hidden_layer_size))

    output = model(seq)
    output_list = np.append(output_list, output.detach().numpy())

scaler.fit_transform(test[:, 1][:, np.newaxis])

real = test_y.detach().numpy().reshape(1, -1)
pred = np.array(output_list)

pred_pandas = pd.DataFrame(scaler.inverse_transform(pred[:, np.newaxis])[:720], columns=["pre_results"])
pred_pandas.to_csv(path + "traffic_lstm_results.csv", index=False)

plt.figure(figsize=[20, 5])
plt.plot(scaler.inverse_transform(real[0][:, np.newaxis])[:720], 'k', label="实际流量", alpha=0.3)
plt.plot(scaler.inverse_transform(pred[:, np.newaxis])[:720], 'r.', label="预测流量")
plt.legend()
plt.grid(alpha=0.3)
plt.title("每小时流量峰值预测（预测步长：一天）")
plt.xlabel("时间（hour）")
plt.ylabel("流量峰值（Gbps）")
plt.savefig("/Users/yuao/PycharmProjects/traffic_prediction/figures/pre_port_max_step_1.png", dpi=600, bbox_inches='tight')
plt.show()
