import pandas as pd
import pickle
import streamlit as st

# Загрузка модели
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Функция обработки CSV
def predict_csv(data, model):
    # Выполнение предсказаний
    predictions = model.predict(data)

    # Добавление предсказаний как нового столбца
    data['Predictions'] = predictions
    return data

# Основной код Streamlit
def main():
    st.title("Прогнозирование, останется ли абонент с текущим оператором связи или нет.")

    # Загрузка модели
    model_path = "mnp.pkl"  # Путь к файлу модели
    model = load_model(model_path)

    # Загрузка CSV файла
    uploaded_file = st.file_uploader("Выберите CSV файл с данными об абонентах", type=["csv"])

    if uploaded_file is not None:
        # Чтение входного CSV
        data = pd.read_csv(uploaded_file)

        # Показать загруженные данные
        st.write("Загруженные данные:")
        st.dataframe(data)

        # Выполнение предсказаний
        if st.button("Сделать предсказания"):
            result_data = predict_csv(data, model)

            # Показать DataFrame с предсказаниями
            st.write("Данные с предсказаниями:")
            st.dataframe(result_data)

            # Сохранение результата в CSV
            output_csv = "output_with_predictions.csv"
            result_data.to_csv(output_csv, index=False)

            # Предложение скачать файл
            st.success("Предсказания выполнены!")
            st.download_button(
                label="Скачать файл с предсказаниями",
                data=result_data.to_csv(index=False).encode('utf-8'),
                file_name='output_with_predictions.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()