import pandas as pd
import pathlib
import warnings
warnings.simplefilter("ignore")

def read():
    month = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
             'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

    dataset_path = pathlib.Path(r'C:\Users\sauth\PycharmProjects\main\dataset')
    df_airbnb = pd.DataFrame()

    for file in dataset_path.iterdir():
        month_name = file.name[:3]
        month_m = month[month_name]
        year = file.name[-8:]
        year = int(year.replace('.csv', ''))

        df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
        df['year'] = year
        df['month'] = month_m
        df_airbnb = df_airbnb.append(df)

    return df_airbnb
