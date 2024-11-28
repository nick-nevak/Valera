import os
import joblib
import pandas as pd


def check():
    print('ZVC2345')


def color_cells(val):
    """ val: значение признака """

    if val == 'float64':
        color = 'red'
    elif val == 'int64':
        color = 'red'
    else:
        color = 'blue'
    return f'color: {color}'


def valeraInfo(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Количесвто уникальных'] = df.nunique()
    info['Количество пропусков'] = df.isna().sum()
    info['Количество значений'] = df.count()
    info['%значений'] = round((df.count() / df.shape[0]) * 100, 2)
    info = info.style.applymap(color_cells, subset=['Тип данных'])
    return info


folder_path = "dumps"  # Specify your folder


def dump_ZV(content, file_name, folder_path="dumps"):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{file_name}.joblib")
    joblib.dump(content, file_path)
    print(f"Saved {file_name} to {file_path}")


def undump_ZV(file_name, folder_path="dumps"):
    file_path = os.path.join(folder_path, f"{file_name}.joblib")
    return joblib.load(file_path)


def get_cols_containing(df, features):
    # Initialize an empty list to store columns
    count_columns = []
    # Iterate over the features to process and collect matching columns
    for feature in features:
        count_columns.extend([col for col in df.columns if feature in col])
    return df[count_columns]
