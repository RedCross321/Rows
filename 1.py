import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

df = pd.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',')

data = df['CO(GT)'].replace(-200, np.nan)
data = data.dropna()
seasonal_diff = data.diff().dropna()
final_diff = seasonal_diff.diff().dropna()

# print(len(final_diff))
# plt.plot(final_diff[6400:])
# plt.show()
# Test stationarity with KPSS
# print("\nKPSS Test for Final Transformed Data:")
# result_kpss = kpss(final_diff[6400:], regression='c')
# print(f'KPSS Statistic: {result_kpss[0]}')
# print(f'p-value: {result_kpss[1]}')
# print('Critical Values:')
# for key, value in result_kpss[3].items():
#     print(f'   {key}: {value}')

# plt.figure(figsize=(15, 8))
# plot_pacf(final_diff, lags=40, ax=plt.gca())
# plt.title('График частичной автокорреляционной функции для NMHC(GT)')
# plt.xlabel('Лаг')
# plt.ylabel('PACF')
# plt.tight_layout()
# plt.show()

new_data = final_diff[6400:].values

model = AutoReg(new_data, lags=20)
model_fit = model.fit()
fittedvalues = model_fit.fittedvalues
# print(model_fit.summary())
plt.figure(figsize=(15, 8))
plt.plot(final_diff[6400:].index, new_data, label='Исходные данные', alpha=0.7)
plt.plot(final_diff[6400:].index[20:], fittedvalues, label='Модель AR(3)', color='red', alpha=0.7)
plt.title('Сравнение исходных данных и модели AR(3)')
plt.ylabel('NMHC(GT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def fit_ar_model(time_series, p):

    X = time_series.copy()
    n = len(X)
    
    if n <= p:
        raise ValueError(f"Длина ряда ({n}) должна быть больше порядка модели ({p})")
    
    X_lagged = np.zeros((n - p, p + 1))
    X_lagged[:, 0] = 1 

    for i in range(1, p + 1):
        X_lagged[:, i] = X[p - i:n - i]

    y = X[p:]
    
    beta = np.linalg.lstsq(X_lagged, y, rcond=None)[0]
    
    a_0 = beta[0]
    a_i = beta[1:]

    return a_0, a_i

def predict_next_value(last_values, a_0, a_i):

    p = len(a_i)
    if len(last_values) != p:
        raise ValueError(f"Для предсказания требуется ровно {p} последних значений")
        
    # X_t = a_0 + sum(a_i * X_{t-i})
    prediction = a_0
    for i in range(p):
        prediction += a_i[i] * last_values[p - 1 - i]
        
    return prediction

def forecast_ar(time_series, p, n_steps):

    X = time_series.copy()
    
    if len(X) < p:
        raise ValueError(f"Длина ряда ({len(X)}) должна быть не меньше порядка модели ({p})")
    
    a_0, a_i = fit_ar_model(X, p)
    
    history = list(X[-p:])
    predictions = []
    
    for _ in range(n_steps):

        next_value = predict_next_value(history, a_0, a_i)
        predictions.append(next_value)
        
        history.pop(0)
        history.append(next_value)
    
    return np.array(predictions), a_0, a_i


N = 50 
p = 100

predictions, a_0, a_i = forecast_ar(new_data, p, N)

plt.figure(figsize=(15, 8))
plt.plot(new_data, label='Исходные данные', alpha=0.7)
plt.plot(range(len(new_data), len(new_data) + N), predictions, label=f'Прогноз AR({p})', color='red', linestyle='--')
plt.axvline(x=len(new_data) - 1, color='green', linestyle=':', label='Начало прогноза')
plt.title(f'Прогноз на {N} точек с помощью авторегрессионной модели AR({p})')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

p_arima = 20 
d_arima = 1
q_arima = 20

arima_data = new_data.copy()
arima_model = ARIMA(arima_data, order=(p_arima, d_arima, q_arima))
arima_model_fit = arima_model.fit()


forecast_steps = N
arima_forecast = arima_model_fit.forecast(steps=forecast_steps)

plt.figure(figsize=(15, 8))
plt.plot(arima_data, label='Исходные данные', alpha=0.7)
plt.plot(range(len(arima_data), len(arima_data) + forecast_steps), arima_forecast, 
         label=f'Прогноз ARIMA({p_arima},{d_arima},{q_arima})', color='blue', linestyle='--')
plt.axvline(x=len(arima_data) - 1, color='green', linestyle=':', label='Начало прогноза')
plt.title(f'Прогноз на {forecast_steps} точек с помощью модели ARIMA({p_arima},{d_arima},{q_arima})')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
