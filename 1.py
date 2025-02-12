import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

# df = pd.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',')
# numeric_columns = df.select_dtypes(include=[np.number]).columns
# result = adfuller(df[numeric_columns[2]].dropna())
# print('adf', result[0])
# print('p-значение: ', result[1])
# print('критическое значение: ', result[4])
# if result[0] > result[4]['5%']:
#     print('Ряд нестационарный')
# else:
#     print('Ряд стационарный')

df = pd.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',')

data = df['NMHC(GT)'].replace(-200, np.nan)
data = data.dropna()

plt.figure(figsize=(15, 8))
pacf_values = pacf(data, nlags=40)
plt.stem(range(len(pacf_values)), pacf_values)
plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
plt.title('График частичной автокорреляционной функции для NMHC(GT)')
plt.xlabel('Лаг')
plt.ylabel('PACF')
plt.tight_layout()
plt.show()

new_data = data.values

model = AutoReg(new_data, lags=3)
model_fit = model.fit()
fittedvalues = model_fit.fittedvalues

plt.figure(figsize=(15, 8))
plt.plot(data.index, new_data, label='Исходные данные', alpha=0.7)
plt.plot(data.index[3:], fittedvalues, label='Модель AR(3)', color='red', alpha=0.7)
plt.title('Сравнение исходных данных и модели AR(3)')
plt.ylabel('NMHC(GT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

ma3 = data.rolling(3).mean()
ma5 = data.rolling(5).mean()
ma10 = data.rolling(10).mean()
ma50 = data.rolling(50).mean()

plt.figure(figsize=(15, 8))
plt.plot(data.index, data.values, label='Исходные данные', alpha=0.3, color='gray')
plt.plot(data.index, ma3, label='MA(3)', linewidth=2)
plt.plot(data.index, ma5, label='MA(5)', linewidth=2)
plt.plot(data.index, ma10, label='MA(10)', linewidth=2)
plt.plot(data.index, ma50, label='MA(50)', linewidth=2)

plt.title('Скользящие средние с разными окнами')
plt.ylabel('NMHC(GT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

model_arma = ARIMA(new_data, order=(3, 0, 3))
model_arma_fit = model_arma.fit()
predictions = model_arma_fit.fittedvalues

plt.figure(figsize=(15, 8))
plt.plot(data.index, data.values, label='Исходные данные', alpha=0.5, color='gray')
plt.plot(data.index, predictions, label='ARMA(3,3) модель', color='blue', linewidth=2)
plt.title('Модель ARMA(3,3) и прогноз')
plt.ylabel('NMHC(GT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()