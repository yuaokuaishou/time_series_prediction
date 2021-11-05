import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

from data_read import read_data, data_prophet

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

path = ""
file_name = "traffic_direction.csv"
file_name_prophet = "traffic_prophet.csv"
file_name_prophet_results = "traffic_prophet_results.csv"

data_in, data_out = read_data(path, file_name)
data = data_out  # 选择in or out

prophet_data = data_prophet(data, path, file_name_prophet)

facebook_engine = Prophet()
facebook_engine.fit(prophet_data)

future = facebook_engine.make_future_dataframe(periods=720, freq="H")

forecast = facebook_engine.predict(future)
forecast.to_csv(path + file_name_prophet_results)

fig1 = facebook_engine.plot(forecast)
fig2 = facebook_engine.plot_components(forecast)


plt.show()
