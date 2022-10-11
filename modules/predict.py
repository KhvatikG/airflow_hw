import dill
import sklearn
import glob
import json
import pandas as pd
import os
# Путь до папки с проэктом:
path = os.environ.get('PROJECT_PATH', '.')

# Дирректория с моделями:
model_dir = (f'{path}/data/models/*.pkl')

# Формируем список путей моделей в дирректории
# и выбираем из него путь к последней модели:
list_models = glob.glob(model_dir)
path_model = (f'{list_models[-1]}')

# Формируем префикс имени файла с предиктами, соответствующий используемой модели:
prefix_name = path_model[-17:-4]
# Путь к файлу с тестовыми данными и путь для сохранения файла с предиктами:
path_test = (f'{path}/data/test/*.json')
path_result = (f'{path}/data/predictions/preds_{prefix_name}.csv')


def predict():
    with open(path_model, 'rb') as file:
        model = dill.load(file)

    # Загружаем список файлов с путями к ним для теста:
    list_test = glob.glob(path_test)

    predicts = dict()
    # Перебираем тестовые файлы из списка
    # и получаем предикт по каждому, сохраняя их в датафрэйм:
    for car in list_test:
        with open(car) as file:
            data = json.load(file)

        pred = model.predict(pd.DataFrame([data]))[0]
        predicts[data['id']] = pred

        predicts_df = pd.DataFrame(predicts.items(), columns=['id', 'pred_class'])

        # Сохраняем фаил с предиктами:
        with open(path_result, 'wb') as file:
            predicts_df.to_csv(file, index=False)


if __name__ == '__main__':
    predict()
