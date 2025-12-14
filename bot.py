import os
import re
import csv
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg") # Установка бэкенда Matplotlib для генерации изображений без GUI
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema # Для поиска локальных экстремумов (минимумов/максимумов)

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA

import tensorflow as tf

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup


# ----------------------------
# КОНФИГУРАЦИЯ
# ----------------------------
# Токен Telegram-бота. Берется из переменных окружения или используется значение по умолчанию.
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
# Путь к файлу для логирования. Берется из переменных окружения или используется значение по умолчанию.
LOG_PATH = os.getenv("LOG_PATH", "logs.csv")
# Горизонт прогнозирования в днях. Берется из переменных окружения или используется значение по умолчанию.
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "30"))

# Регулярное выражение для проверки корректности тикера акции.
TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-]{0,15}$")


# ----------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ----------------------------
# Функция для расчета среднеквадратичной ошибки (RMSE)
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# Функция для безопасной обработки тикера, приведения к верхнему регистру и валидации.
def safe_ticker(text: str) -> str:
    t = (text or "").strip().upper()
    if not TICKER_RE.match(t):
        raise ValueError("Некорректный тикер. Пример: AAPL, MSFT.")
    return t


# Функция для безопасной обработки суммы инвестиции, преобразования в float и валидации.
def safe_amount(text: str) -> float:
    x = float((text or "").replace(",", ".").strip())
    if x <= 0:
        raise ValueError("Сумма должна быть > 0.")
    return x


# Функция для добавления записи в CSV-файл логов.
def append_log(row: dict):
    exists = os.path.exists(LOG_PATH)
    row = dict(row)
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: # Если файл не существует, записываем заголовок
            w.writeheader()
        w.writerow(row)


# Функция для загрузки исторических цен акций с помощью yfinance.
def load_prices(ticker: str, period: str = "2y") -> pd.DataFrame:
    # period="2y" поддерживается в примерах/документации yfinance
    df = yf.Ticker(ticker).history(period=period) # Загрузка данных
    if df is None or df.empty:
        raise ValueError("Нет данных по тикеру (проверьте символ).")
    # Переименование колонок и выбор нужных
    df = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    df = df[["ds", "y"]].dropna()
    df["ds"] = pd.to_datetime(df["ds"]) # Преобразование 'Date' в datetime
    df["y"] = df["y"].astype(float) # Преобразование 'Close' в float
    df = df.sort_values("ds").reset_index(drop=True) # Сортировка по дате
    return df


# Функция для построения и сохранения графика прогноза.
def plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, out_path: str):
    plt.figure(figsize=(11, 5)) # Создание новой фигуры
    plt.plot(history["ds"], history["y"], label="History", linewidth=2) # Исторические данные
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast(30d)", linewidth=2) # Прогнозные данные
    plt.grid(True, alpha=0.3) # Добавление сетки
    plt.title("Stock price forecast") # Заголовок графика
    plt.xlabel("Date") # Подпись оси X
    plt.ylabel("Price") # Подпись оси Y
    plt.legend() # Отображение легенды
    plt.tight_layout() # Автоматическая корректировка параметров подграфиков для плотного размещения
    plt.savefig(out_path, dpi=160) # Сохранение графика в файл
    plt.close() # Закрытие фигуры для освобождения памяти


# ----------------------------
# ФИЧИ ДЛЯ ML-МОДЕЛИ
# ----------------------------
# Функция для создания лаговых признаков и признаков скользящих средних/стандартных отклонений.
def make_lag_features(df: pd.DataFrame, lags=(1, 2, 3, 5, 10, 20), windows=(5, 10, 20)) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out["y"].shift(lag) # Создание лаговых признаков
    for w in windows:
        out[f"roll_mean_{w}"] = out["y"].shift(1).rolling(w).mean() # Скользящее среднее
        out[f"roll_std_{w}"] = out["y"].shift(1).rolling(w).std() # Скользящее стандартное отклонение
    out["dow"] = out["ds"].dt.dayofweek # День недели
    out["month"] = out["ds"].dt.month # Месяц
    out = out.dropna().reset_index(drop=True) # Удаление строк с NaN и сброс индекса
    return out


# Функция для разделения временного ряда на обучающую и тестовую выборки.
def split_time(df: pd.DataFrame, test_size: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_size)) # Расчет точки разделения
    return df.iloc[:cut].copy(), df.iloc[cut:].copy() # Возврат обучающей и тестовой выборок


# ----------------------------
# МОДЕЛИ
# ----------------------------
# Функция для обучения и прогнозирования с помощью RandomForestRegressor.
def fit_predict_rf(df: pd.DataFrame, horizon: int = 30):
    feat = make_lag_features(df) # Создание признаков
    train_df, test_df = split_time(feat, 0.2) # Разделение данных

    X_train = train_df.drop(columns=["ds", "y"]) # Признаки для обучения
    y_train = train_df["y"].values # Целевая переменная для обучения
    X_test = test_df.drop(columns=["ds", "y"]) # Признаки для тестирования
    y_test = test_df["y"].values # Целевая переменная для тестирования

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_depth=12
    )
    model.fit(X_train, y_train) # Обучение модели
    pred_test = model.predict(X_test) # Прогнозирование на тестовой выборке
    score = rmse(y_test, pred_test) # Расчет RMSE

    # Рекурсивный прогноз на будущее
    hist = df.copy().reset_index(drop=True)
    future = []
    for _ in range(horizon):
        tmp = make_lag_features(hist) # Создание признаков для последнего дня истории
        last_X = tmp.iloc[-1:].drop(columns=["ds", "y"]) # Признаки последнего дня
        y_next = float(model.predict(last_X)[0]) # Прогноз следующего значения
        next_date = hist["ds"].iloc[-1] + pd.Timedelta(days=1) # Следующая дата
        # Добавление прогноза в историю для следующей итерации
        hist = pd.concat([hist, pd.DataFrame({"ds": [next_date], "y": [y_next]})], ignore_index=True)
        future.append((next_date, y_next)) # Сохранение прогноза

    forecast = pd.DataFrame(future, columns=["ds", "yhat"]) # Формирование DataFrame прогноза
    return model, score, forecast


# Функция для обучения и прогнозирования с помощью модели ARIMA.
def fit_predict_arima(df: pd.DataFrame, horizon: int = 30):
    series = df[["ds", "y"]].copy().reset_index(drop=True)
    train_df, test_df = split_time(series, 0.2) # Разделение данных

    y_train = train_df["y"].values # Обучающая выборка
    y_test = test_df["y"].values # Тестовая выборка

    model = ARIMA(y_train, order=(5, 1, 0)) # Создание модели ARIMA
    fitted = model.fit() # Обучение модели

    pred_test = fitted.forecast(steps=len(y_test)) # Прогноз на тестовой выборке
    score = rmse(y_test, pred_test) # Расчет RMSE

    pred_future = fitted.forecast(steps=horizon) # Прогноз на будущее
    # Создание дат для будущего прогноза
    future_dates = pd.date_range(start=series["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.DataFrame({"ds": future_dates, "yhat": pred_future}) # Формирование DataFrame прогноза
    return fitted, score, forecast


# Функция для создания последовательностей данных для LSTM.
def make_sequences(arr: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i]) # Входная последовательность
        y.append(arr[i]) # Целевое значение
    return np.array(X), np.array(y)


# Функция для обучения и прогнозирования с помощью модели LSTM.
def fit_predict_lstm(df: pd.DataFrame, horizon: int = 30, lookback: int = 30, epochs: int = 12):
    series = df[["ds", "y"]].copy().reset_index(drop=True)
    train_df, test_df = split_time(series, 0.2) # Разделение данных

    scaler = MinMaxScaler() # Инициализация нормализатора
    train_scaled = scaler.fit_transform(train_df[["y"]]).astype(np.float32) # Нормализация обучающей выборки
    test_scaled = scaler.transform(test_df[["y"]]).astype(np.float32) # Нормализация тестовой выборки

    X_train, y_train = make_sequences(train_scaled, lookback) # Создание последовательностей для обучения
    X_test, y_test = make_sequences(test_scaled, lookback) # Создание последовательностей для тестирования

    if len(X_train) < 10 or len(X_test) < 5:
        raise ValueError("Недостаточно данных для LSTM (после разбиения/окна).")

    # Изменение формы данных для LSTM слоя
    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    tf.keras.utils.set_random_seed(42) # Установка seed для воспроизводимости

    model = tf.keras.Sequential([ # Создание последовательной модели Keras
        tf.keras.layers.Input(shape=(lookback, 1)), # Входной слой
        tf.keras.layers.LSTM(64, return_sequences=True), # Первый LSTM слой
        tf.keras.layers.Dropout(0.2), # Dropout для регуляризации
        tf.keras.layers.LSTM(32), # Второй LSTM слой
        tf.keras.layers.Dense(1), # Выходной слой
    ])
    model.compile(optimizer="adam", loss="mse") # Компиляция модели
    # Обучение модели
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=32, verbose=0)

    pred_test = model.predict(X_test, verbose=0) # Прогноз на тестовой выборке
    pred_test_inv = scaler.inverse_transform(pred_test).ravel() # Обратное масштабирование прогнозов
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel() # Обратное масштабирование истинных значений
    score = rmse(y_test_inv, pred_test_inv) # Расчет RMSE

    # Рекурсивный прогноз на будущее
    full_scaled = scaler.transform(series[["y"]]).astype(np.float32).ravel().tolist() # Нормализованные все данные
    future_vals = []
    for _ in range(horizon):
        # Подготовка входной последовательности для прогноза
        x = np.array(full_scaled[-lookback:], dtype=np.float32).reshape((1, lookback, 1))
        y_next_scaled = float(model.predict(x, verbose=0)[0][0]) # Прогноз следующего значения
        full_scaled.append(y_next_scaled) # Добавление прогноза в масштабированный ряд
        y_next = float(scaler.inverse_transform([[y_next_scaled]])[0][0]) # Обратное масштабирование
        future_vals.append(y_next) # Сохранение прогноза

    # Создание дат для будущего прогноза
    future_dates = pd.date_range(start=series["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.DataFrame({"ds": future_dates, "yhat": future_vals}) # Формирование DataFrame прогноза
    return model, score, forecast


# Функция для запуска всех моделей и выбора лучшей.
def run_all_models(df: pd.DataFrame, horizon: int = 30):
    results = []

    # Запуск RandomForest
    rf_model, rf_score, rf_fc = fit_predict_rf(df, horizon=horizon)
    results.append(("RandomForest", rf_model, rf_score, rf_fc))

    # Запуск ARIMA
    ar_model, ar_score, ar_fc = fit_predict_arima(df, horizon=horizon)
    results.append(("ARIMA", ar_model, ar_score, ar_fc))

    try:
        # Запуск LSTM
        lstm_model, lstm_score, lstm_fc = fit_predict_lstm(df, horizon=horizon)
        results.append(("LSTM", lstm_model, lstm_score, lstm_fc))
    except Exception as e:
        # Если LSTM не запустился (например, из-за недостатка данных/ресурсов), пропускаем его.
        results.append(("LSTM (skipped)", None, float("inf"), pd.DataFrame({"ds": [], "yhat": []})))

    # Выбор лучшей модели по наименьшему RMSE
    best = min(results, key=lambda x: x[2])
    return results, best


# ----------------------------
# СТРАТЕГИЯ
# ----------------------------
# Функция для определения сигналов покупки/продажи на основе локальных экстремумов прогноза.
def extrema_signals(forecast: pd.DataFrame, order: int = 3) -> pd.DataFrame:
    y = forecast["yhat"].values.astype(float)
    # Проверка достаточного количества данных для поиска экстремумов
    if len(y) < (2 * order + 1):
        return pd.DataFrame(columns=["ds", "yhat", "signal"])

    max_idx = argrelextrema(y, np.greater, order=order)[0] # Индексы локальных максимумов
    min_idx = argrelextrema(y, np.less, order=order)[0] # Индексы локальных минимумов

    buys = forecast.iloc[min_idx][["ds", "yhat"]].copy() # Сигналы покупки (на минимумах)
    sells = forecast.iloc[max_idx][["ds", "yhat"]].copy() # Сигналы продажи (на максимумах)
    buys["signal"] = "BUY"
    sells["signal"] = "SELL"
    # Объединение и сортировка сигналов
    sig = pd.concat([buys, sells], ignore_index=True).sort_values("ds").reset_index(drop=True)
    return sig


# Функция для симуляции прибыли на основе торговых сигналов.
def simulate_profit(forecast: pd.DataFrame, investment: float, order: int = 3):
    sig = extrema_signals(forecast, order=order) # Получение торговых сигналов

    cash = float(investment) # Начальный капитал
    shares = 0.0 # Количество акций
    last_action = None # Последнее действие (для предотвращения двойных действий)

    for _, r in sig.iterrows():
        price = float(r["yhat"])
        # Условие покупки: есть деньги, акций нет, и последнее действие не было покупкой
        if r["signal"] == "BUY" and cash > 0 and last_action != "BUY":
            shares = cash / price # Покупаем акции
            cash = 0.0 # Обнуляем кэш
            last_action = "BUY"
        # Условие продажи: есть акции, и последнее действие не было продажей
        elif r["signal"] == "SELL" and shares > 0 and last_action != "SELL":
            cash = shares * price # Продаем акции
            shares = 0.0 # Обнуляем акции
            last_action = "SELL"

    last_price = float(forecast["yhat"].iloc[-1]) # Цена на последний день прогноза
    # Расчет конечной стоимости портфеля
    final_value = cash if shares == 0 else shares * last_price
    profit = final_value - float(investment) # Расчет прибыли
    return sig, final_value, profit


# ----------------------------
# БОТ (aiogram 3)
# ----------------------------
# Класс для управления состояниями пользователя в FSM (Finite State Machine).
class Form(StatesGroup):
    ticker = State() # Состояние ожидания тикера
    amount = State() # Состояние ожидания суммы инвестиции


# Основная асинхронная функция для запуска бота.
async def main():
    if not BOT_TOKEN:
        raise RuntimeError("Нужно задать переменную окружения BOT_TOKEN (или в .env при запуске через systemd).")

    bot = Bot(BOT_TOKEN) # Инициализация бота
    dp = Dispatcher() # Инициализация диспетчера

    # Обработчик команды /start
    @dp.message(CommandStart())
    async def start(m: Message, state: FSMContext):
        await state.clear() # Очистка состояния пользователя
        await m.answer("Введите тикер компании (например, AAPL):") # Запрос тикера
        await state.set_state(Form.ticker) # Установка состояния 'ticker'

    # Обработчик состояния 'ticker' (после ввода тикера).
    @dp.message(Form.ticker)
    async def handle_ticker(m: Message, state: FSMContext):
        try:
            ticker = safe_ticker(m.text) # Валидация тикера
        except Exception as e:
            await m.answer(f"{e}\nВведите тикер ещё раз:") # Сообщение об ошибке и повторный запрос
            return
        await state.update_data(ticker=ticker) # Сохранение тикера в состоянии
        await m.answer("Введите сумму для условной инвестиции (например, 10000):") # Запрос суммы
        await state.set_state(Form.amount) # Установка состояния 'amount'

    # Обработчик состояния 'amount' (после ввода суммы).
    @dp.message(Form.amount)
    async def handle_amount(m: Message, state: FSMContext):
        data = await state.get_data() # Получение данных из состояния
        ticker = data["ticker"]
        user_id = m.from_user.id if m.from_user else None

        try:
            amount = safe_amount(m.text) # Валидация суммы
        except Exception as e:
            await m.answer(f"{e}\nВведите сумму ещё раз:") # Сообщение об ошибке и повторный запрос
            return

        await m.answer("Загружаю данные и обучаю модели (это может занять 1–2 минуты)...") # Уведомление пользователя

        img_path = f"/tmp/forecast_{ticker}_{user_id}.png" # Путь для сохранения графика
        try:
            hist = load_prices(ticker, period="2y") # Загрузка исторических данных
            all_results, best = run_all_models(hist, horizon=FORECAST_HORIZON) # Запуск моделей и выбор лучшей
            best_name, _, best_rmse, forecast = best

            if forecast is None or forecast.empty:
                raise ValueError("Не удалось построить прогноз (проверьте модели).")

            # Вывод названия и RMSE каждой модели
            model_performance_text = "\nРезультаты моделей:\n"
            for name, model_obj, rmse_score, forecast_df in all_results:
                model_performance_text += f"  Модель: {name:<15} RMSE: {rmse_score:.4f}\n"
            await m.answer(model_performance_text)

            current_price = float(hist["y"].iloc[-1]) # Текущая цена
            future_last = float(forecast["yhat"].iloc[-1]) # Прогнозируемая цена на последний день
            pct_change = (future_last - current_price) / current_price * 100.0 # Процентное изменение

            signals, final_value, profit = simulate_profit(forecast, investment=amount, order=3) # Симуляция прибыли

            plot_forecast(hist, forecast, img_path) # Построение и сохранение графика

            # FSInputFile — стандартный способ отправки локального файла в aiogram 3 [web:23]
            # Отправка графика с описанием
            await m.answer_photo(
                photo=FSInputFile(img_path),
                caption=(
                    f"Тикер: {ticker}\n"
                    f"Лучшая модель: {best_name}\n"
                    f"RMSE: {best_rmse:.4f}\n"
                    f"Прогноз на {FORECAST_HORIZON} дней: {pct_change:.2f}% к последнему дню"
                ),
            )

            # Формирование текста рекомендаций (компактная сводка сигналов)
            if signals.empty:
                rec_text = "Сигналы BUY/SELL не найдены (ряд слишком гладкий)."
            else:
                head = signals.head(12)
                lines = [f"{r.ds.date()} — {r.signal} @ {float(r.yhat):.2f}" for r in head.itertuples()]
                rec_text = "\n".join(lines)

            # Отправка рекомендаций и результатов симуляции
            await m.answer(
                "Рекомендации (до 12 первых сигналов):\n"
                f"{rec_text}\n\n"
                f"Условная итоговая стоимость: {final_value:.2f}\n"
                f"Условная прибыль: {profit:.2f}\n\n"
                "Дисклеймер: результаты учебные, не финсовет."
            )

            # Логирование результатов
            append_log({
                "user_id": user_id,
                "ticker": ticker,
                "amount": round(amount, 2),
                "best_model": best_name,
                "rmse": round(float(best_rmse), 6),
                "profit": round(float(profit), 2),
                "final_value": round(float(final_value), 2),
            })

        except Exception as e:
            await m.answer(f"Ошибка: {e}\nПопробуйте другой тикер или повторите позже.") # Сообщение об ошибке
        finally:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path) # Удаление временного файла графика
            except Exception:
                pass
            await state.clear() # Очистка состояния после завершения обработки

    await dp.start_polling(bot) # Запуск бота


if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop() # Пытаемся получить текущий запущенный цикл событий
    except RuntimeError:
        loop = None # Если цикла нет, устанавливаем loop в None

    if loop and loop.is_running():
        # Если цикл уже запущен (например, в Colab), планируем main как задачу
        loop.create_task(main())
    else:
        # В противном случае запускаем main в новом цикле событий
        asyncio.run(main())