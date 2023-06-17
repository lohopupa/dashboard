import pickle
from random import random
import math

import pandas as pd
import numpy as np
import streamlit as st


def description_page(st):
    st.title("Описание задачи и данных")
    st.write("Выберите страницу слева")

    st.header("Описание задачи")
    st.markdown("""Задачей данного проекта является распознавание пользователя по его движению курсором(курсорному почерку).

Для классификации разных пользователей для каждого должна быть обучена модель способная отличить его почерк от почерка других пользователей
Для примера была обучена модель для одного пользователя и с её помощью можно узнать принадлежит ли запись траектории движения этому пользователю.""")

    st.header("Описание данных")
    st.markdown("""Предоставленные данные:

* avarageX - Среднее перемещение по оси Х
* avarageY - Среднее перемещение по оси У
* stdDevX - Стандартное отклонение по оси Х
* stdDevY - Стандартное отклонение по оси У
* length - Длин траектории
* distance - Расстояние между началом и концом траектории
* averageVelocity - Средняя скоркость 
* maxVelocity - Максимальная скорость 
* angleDeviation - Угол между направлением движения и направлением на цель
* timeWithoutMoving - Время без изменеия координат(зависит от дельты)
* averageAcceleration - Среднее ускорение
* maxAcceleration - Максимальное ускорение
* commonTimeOfAcceleration - Общее время ускорения
* commonTimeOfDecceleration - Общее время замедления
""")


def model_query_rmse(st, model, data):
    st.header("Корень из среднеквадратичной ошибки")
    rmse = 3006.55  # Костыль! Заменить на настоящий подсчёт метрики
    st.write(f"{rmse}")


def model_query_n_random(st, model, data, n):
    st.header(f"Первые {n} предсказанных значений")
    max_id = 182
    for i in range(n):
        idx = math.floor(random() * max_id + 1)
        st.write(f"Предсказанное значение для класса {idx}: {predict(model, data, idx)}")
    st.button('Обновить')


def model_query_user_query(st, model, data):
    max_id = 182
    idx = st.number_input(f"Категория  от 1 до {max_id}", 1, max_id)

    if st.button('Предсказать'):
        st.write(f"Предсказанное значение: {predict(model, data, idx)}")
    else:
        pass


def predict(model, data, class_id):
    d = data[data["class"] == class_id]
    t = 0
    for i in range(len(d)):
        p = model.predict(np.array(d.drop(columns=["class"]).iloc[i]).reshape((1, -1)))
        if p:
            t+=1
    return t/len(d)>.5



@st.cache_data
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file)
    return df

n = 10
model_pages = {
    # "RMSE": model_query_rmse,
    f"{n} случайных предсказанных значений": lambda *args : model_query_n_random(*args, n),
    "Пользовательский пример": model_query_user_query
}

def model_query_page(st):
    model = load_model("./model_dumps/knn.sav")
    data = load_test_data("data/dataset.csv")

    st.title("Запрос к модели(данная модель обучена на отделение 10 класа от остальных)")
    st.write("Выберите страницу слева")
    request = st.selectbox(
        "Выберите запрос",
        list(model_pages)
    )

    model_pages[request](st, model, data)
    
pages = {
    "Описание задачи и данных": description_page,
    "Запрос к модели": model_query_page,
}
