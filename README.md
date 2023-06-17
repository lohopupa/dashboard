# Демо-приложение на `streamlit`
### Приложение основано на [примере](https://github.com/otverskoj/streamlit-demo)

## Описание

Представлен полный код простого приложения, построенного с использованием библиотеки `streamlit`. Показано использование базовых виджетов, а также работа с кешированными данными.

Описание структуры проекта:
- директория `model_dumps` содержит файлы сериализованных моделей (`.pkl`, `.sav` и т.д.)
- директория `data` содержит файлы с предобработанными данными для моделей
- файл `app.py` содержит основной код приложения

## Установка зависимостей

Для установки необходимых библиотек, запустите в терминале/консоли следующий фрагмент кода:

```shell
pip install -r requirements.txt
```

## Запуск приложения

Для запуска приложения выполните команду:

```shell
streamlit run app.py
```
