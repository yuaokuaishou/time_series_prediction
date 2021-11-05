import pandas as pd


def read_data(path, file_name):
    data = pd.read_csv(path + file_name)
    data_in, data_out = data.loc[data["direction"] == "in"], data.loc[data["direction"] == "out"]
    return data_in, data_out


def data_prophet(data, path, file_name_prophet):
    data = data[["p_date", "p_hourmin", "max_rate"]]
    data.reset_index(inplace=True)
    data = data.drop(["index"], axis=1)
    data = data[:8016]  # 8016:预测10月，8759：预测11月

    data["ds"] = data["p_date"]
    data["y"] = data["max_rate"]
    data["hour"] = data["p_hourmin"]

    i = 0
    while i <= 8016:  # 8016:预测10月，8759：预测11月
        data["hour"][i + 0] = "00:00:00"
        data["hour"][i + 1] = "01:00:00"
        data["hour"][i + 2] = "02:00:00"
        data["hour"][i + 3] = "03:00:00"
        data["hour"][i + 4] = "04:00:00"
        data["hour"][i + 5] = "05:00:00"
        data["hour"][i + 6] = "06:00:00"
        data["hour"][i + 7] = "07:00:00"
        data["hour"][i + 8] = "08:00:00"
        data["hour"][i + 9] = "09:00:00"
        data["hour"][i + 10] = "10:00:00"
        data["hour"][i + 11] = "11:00:00"
        data["hour"][i + 12] = "12:00:00"
        data["hour"][i + 13] = "13:00:00"
        data["hour"][i + 14] = "14:00:00"
        data["hour"][i + 15] = "15:00:00"
        data["hour"][i + 16] = "16:00:00"
        data["hour"][i + 17] = "17:00:00"
        data["hour"][i + 18] = "18:00:00"
        data["hour"][i + 19] = "19:00:00"
        data["hour"][i + 20] = "20:00:00"
        data["hour"][i + 21] = "21:00:00"
        data["hour"][i + 22] = "22:00:00"
        data["hour"][i + 23] = "23:00:00"
        i += 24

    for i in range(len(data["p_date"])):

        data["ds"][i] = str(data["p_date"][i])[:4] + "-" + str(data["p_date"][i])[4:6] + "-" + str(data["p_date"][i])[6:8] + " " + data["hour"][i]
        print(data["ds"][i])
    data = data.drop(["p_date", "p_hourmin", "hour", "max_rate"], axis=1)
    data.to_csv(path + file_name_prophet, index=0)
    return data
