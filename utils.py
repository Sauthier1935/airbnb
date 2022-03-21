import numpy as np
import pandas as pd
import graphics
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

class Utils:

    def print_df_types(df):
        print(df.dtypes)
        print('-' * 20 + 'exemplo tipo' + '-' * 20)
        print(df.iloc[0])
        print(df.shape)

    def str_attributes_to_int(df):
        df =df.str.replace('$', '')
        df = df.str.replace(',', '')
        df = df.astype(np.float32, copy=False)
        return df

    def labelencoder__attributes_to_int(df,column_name):
        labelencoder = LabelEncoder()
        df[column_name] = labelencoder.fit_transform(df[column_name])
        return df

    def limit(column_name):
        q1 = column_name.quantile(0.25)
        q3 = column_name.quantile(0.75)
        amplitude = q3 - q1
        return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

    def remove_outliers(df, column_name):
        num_lines = df.shape[0]
        lim_inf, lim_sup = Utils.limit(df[column_name])
        df = df.loc[(df[column_name] >= lim_inf) & (df[column_name] <= lim_sup), :]
        return df, num_lines - df.shape[0]

    def split_data(df, n_split):
        df = resample(df, n_samples = n_split, random_state = 42)
        return df

    def selection(df, target_column):
        target = df[target_column].copy()
        return target, df

    def preparation(df, target_column):
        attributes = df.copy()
        attributes = attributes.drop(target_column,axis = 1 ,inplace = False)
        target = df[target_column].copy()
        attributes, target = SMOTE().fit_resample(attributes, target)
        attributes, target = shuffle(attributes, target)
        attributes = pd.DataFrame(attributes)
        target = pd.DataFrame(target)

        print(attributes.info())

        attributes.shape, target.shape

        return target.values, attributes.values

    def analysis_continuous_attributes(df, column_name):
        graphics.Graphics_utils.box_diagram(df[column_name])
        graphics.Graphics_utils.histogram(df[column_name])
        df, removed_lines = Utils.remove_outliers(df,column_name)
        print('{} linhas removidas'.format(removed_lines))
        graphics.Graphics_utils.histogram(df[column_name])
        return df

    def analysis_discrete_attributes(df, column_name):
        graphics.Graphics_utils.box_diagram(df[column_name])
        graphics.Graphics_utils.barr_graph(df[column_name])
        df, removed_lines = Utils.remove_outliers(df, column_name)
        print('{} linhas removidas'.format(removed_lines))
        graphics.Graphics_utils.box_diagram(df[column_name])
        graphics.Graphics_utils.barr_graph(df[column_name])
        return df

    def analysis_text_attributes(df, column_name, num):
        print(df[column_name].value_counts())
        graphics.Graphics_utils.text_attribute_graph(df, column_name)
        table_property_types = df[column_name].value_counts()
        group_columns = []
        for table_type in table_property_types.index:
          if table_property_types[table_type] < num:
            group_columns.append(table_type)
        for table_type in group_columns:
          df.loc[df[column_name] == table_type,column_name] = 'Others'
        graphics.Graphics_utils.text_attribute_graph(df, column_name)
        return df

    def create_new_attribute(col):
        new_cat = []
        for line in col:
            if line <= max((col) * 0.25):
                value = 0
            elif line <= max((col) * 0.5):
                value = 1
            elif line <= max((col) * 0.75):
                value = 2
            else:
                value = 3
            new_cat.extend([value])
        return new_cat