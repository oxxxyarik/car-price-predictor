import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

MODEL_PATH = "car_price_model.cbm"


@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model


model = load_model()

st.set_page_config(page_title="Оценка стоимости автомобиля", page_icon="🚗")

st.title("Оценка стоимости автомобиля")
st.write("Заполните характеристики автомобиля.")

brand = st.text_input("Марка", "Лада")
name = st.text_input("Модель", "2110")
body_type = st.selectbox(
    "Тип кузова",
    ["Хэтчбек 5 дв.", "Хэтчбек 3 дв.", "Лифтбек", "Джип 3 дв.", "Джип 5 дв.", "Седан", "Универсал", "Минивэн", "Пикап", "Открытый", "Купе"]
)
color = st.text_input("Цвет", "Синий")

fuel_type = st.selectbox(
    "Топливо",
    ["Бензин", "Дизель", "Гибрид", "Электро", "Газ"]
)

transmission = st.selectbox(
    "Коробка передач",
    ["Механика", "Автомат", "Робот", "Вариатор"]
)

location = st.text_input("Город", "Чебоксары")

year = st.number_input(
    "Год выпуска",
    min_value=1970,
    max_value=2026,
    value=2010
)

car_age = 2026 - year

default_mileage = int(car_age * 15000)

mileage = st.number_input(
    "Пробег (км)",
    value=default_mileage
)

power = st.number_input(
    "Мощность (л.с.)",
    value=100
)

engine_vol = st.number_input(
    "Объем двигателя",
    value=1.6,
    step=0.1
)

if st.button("Рассчитать стоимость"):

    km_per_year = mileage / (car_age + 1)
    desc_len = 250

    df = pd.DataFrame({
        "brand": [brand],
        "name": [name],
        "bodyType": [body_type],
        "color": [color],
        "fuelType": [fuel_type],
        "transmission": [transmission],
        "power": [power],
        "engine_vol": [engine_vol],
        "year": [year],
        "mileage": [mileage],
        "car_age": [car_age],
        "km_per_year": [km_per_year],
        "desc_len": [desc_len],
        "location": [location]
    })

    model_features = model.feature_names_
    df = df[model_features]

    prediction_log = model.predict(df)[0]
    prediction = np.expm1(prediction_log)

    min_price = prediction * 0.84
    max_price = prediction * 1.16

    st.success("Оценка завершена!")

    st.metric(
        "Средняя рыночная цена",
        f"{prediction:,.0f} ₽".replace(",", " ")
    )

    st.info(
        f"Диапазон цены:\n\n"
        f"**{min_price:,.0f} ₽ — {max_price:,.0f} ₽**".replace(",", " ")
    )