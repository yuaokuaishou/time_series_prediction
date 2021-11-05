import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

path = "/Users/yuao/Downloads/"

data_real = pd.read_csv(path + "traffic_direction.csv")
data_real = data_real.loc[data_real["direction"] == "in"]
data_real = data_real[-720:]["max_rate"]
data_real = data_real.values.tolist()

data_prophet = pd.read_csv(path + "traffic_prophet_results.csv")
data_prophet = data_prophet[8016:]["yhat"]
data_prophet = data_prophet.values.tolist()

data_lstm = pd.read_csv(path + "traffic_lstm_results.csv")
data_lstm = data_lstm.values.tolist()

plt.figure(figsize=[30, 5])
plt.plot(data_real, 'k', label="Real Data", alpha=0.3)
plt.plot(data_prophet, 'b', label="Meta", alpha=0.8)
plt.plot(data_lstm, 'r', label="LSTM")
plt.legend()
plt.title("每小时流量峰值预测（预测步长：一个月）")
plt.xlabel("时间（hour）")
plt.ylabel("流量峰值（Gbps）")
plt.savefig("/Users/yuao/PycharmProjects/traffic_prediction/figures/compare.png", dpi=600, bbox_inches='tight')
plt.show()