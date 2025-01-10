import pandas as pd
import pickle


# Загрузка модели
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# Функция обработки CSV
def predict_csv(input_csv_path, output_csv_path, model):
    # Чтение входного CSV
    data = pd.read_csv(input_csv_path)

    # Выполнение предсказаний
    predictions = model.predict(data)

    # Добавление предсказаний как нового столбца
    data['Predictions'] = predictions

    # Сохранение результата в новый CSV
    data.to_csv(output_csv_path, index=False)
    print(f"Результат сохранён в файл: {output_csv_path}")


# Основной код
if __name__ == "__main__":
    # Укажите пути к входному CSV и выходному файлу
    input_csv_path = "df_mini.csv"  # Путь к исходному CSV
    output_csv_path = "output_with_predictions.csv"  # Путь к файлу с предсказаниями
    model_path = "mnp.pkl"  # Путь к файлу модели

    # Загрузка модели
    model = load_model(model_path)

    # Обработка CSV
    predict_csv(input_csv_path, output_csv_path, model)
