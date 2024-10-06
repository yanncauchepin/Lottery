import os
import pandas as pd
import numpy as np

csv_options = {
    'sep': ';',
    'quotechar': '"',
    'quoting': 0,
    'escapechar': None,
    'encoding': 'latin1',
    'index_col': False
    }

def int_from_date(date):
    if len(str(date).split('/'))==1:
        return date
    else:
        day, month, year = str(date).split('/')
        if len(year)==2:
            year = '20' + year
        day = int(day); month = int(month); year = int(year)
        result = year*10000 + month*100 + day
        return result

def extract_informations_loto(original_dataframe):
    data = {}
    for i, draw in original_dataframe.iterrows():
        print(draw['date_de_tirage'])
        number = int_from_date(draw['date_de_tirage'])
        ball_1 = int(draw['boule_1'])
        ball_2 = int(draw['boule_2'])
        ball_3 = int(draw['boule_3'])
        ball_4 = int(draw['boule_4'])
        ball_5 = int(draw['boule_5'])
        star = int(draw['numero_chance'])
        data[number] = {
            'ball_1' : ball_1, 
            'ball_2' : ball_2, 
            'ball_3' : ball_3, 
            'ball_4' : ball_4, 
            'ball_5' : ball_5, 
            'star' : star
            }
    return data

def extract_informations_euromillions(original_dataframe):
    data = {}
    for i, draw in original_dataframe.iterrows():
        number = int_from_date(draw['date_de_tirage'])
        ball_1 = int(draw['boule_1'])
        ball_2 = int(draw['boule_2'])
        ball_3 = int(draw['boule_3'])
        ball_4 = int(draw['boule_4'])
        ball_5 = int(draw['boule_5'])
        star_1 = int(draw['etoile_1'])
        star_2 = int(draw['etoile_2'])
        data[number] = {
            'ball_1' : ball_1, 
            'ball_2' : ball_2, 
            'ball_3' : ball_3, 
            'ball_4' : ball_4, 
            'ball_5' : ball_5, 
            'star_1' : star_1, 
            'star_2' : star_2
            }
    return data

def build_dataframe(lottery, root_data, path_to_dataframe):
    data = {}
    for file_dataframe in os.listdir(root_data):
        print(file_dataframe)
        file_dataframe = os.path.join(root_data, file_dataframe)
        df = pd.read_csv(file_dataframe, **csv_options)
        dict_info = eval(f'extract_informations_{lottery}')(df)
        data.update(dict_info)
    dataframe = pd.DataFrame(data=data)
    dataframe = dataframe.sort_index(axis=1, ascending=True)
    dataframe = dataframe.T
    dataframe.to_csv(path_to_dataframe)
    return dataframe

def __sort_dataframe_by_integer_index(df):
    integer_values = [int(index.split('__')[0]) for index in df.index]
    sorted_indices = [index for _, index in sorted(zip(integer_values, df.index))]
    sorted_df = df.loc[sorted_indices]
    return sorted_df

def one_hot_encoding(df, lottery):
    print("One hot encoding")
    all_df = pd.DataFrame()
    ball_df = pd.DataFrame()
    star_df = pd.DataFrame()
    for column in df.columns:
        df[column] = df[column].astype('category')
        one_hot_encoded = pd.get_dummies(df[column])
        one_hot_encoded.index = [f'{row}__{column}' for row in one_hot_encoded.index]
        if column in ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']:
            ball_df = pd.concat([ball_df, one_hot_encoded], axis=0)
        else:
            star_df = pd.concat([star_df, one_hot_encoded], axis=0)
        all_df = pd.concat([all_df, one_hot_encoded], axis=0)
    ball_df = __sort_dataframe_by_integer_index(ball_df)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/all_one_hot_ball_{lottery}.csv')
    ball_df.to_csv(path_to_dataframe)
    star_df = __sort_dataframe_by_integer_index(star_df)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/all_one_hot_star_{lottery}.csv')
    star_df.to_csv(path_to_dataframe)
    all_df = __sort_dataframe_by_integer_index(all_df)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/all_one_hot_{lottery}.csv')
    all_df.to_csv(path_to_dataframe)
    return {'all_df': all_df, 'ball_df': ball_df, 'star_df': star_df}

def original_encoding(df, lottery) :
    print("Original encoding")
    all_df = pd.DataFrame()
    ball_df = pd.DataFrame()
    star_df = pd.DataFrame()
    for column in df.columns:
        df[column] = df[column].astype('category')
        original_encoded = df[column]
        original_encoded.index = [f'{row}__{column}' for row in original_encoded.index]
        if column in ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']:
            ball_df = pd.concat([ball_df, original_encoded], axis=0)
        else:
            star_df = pd.concat([star_df, original_encoded], axis=0)
        all_df = pd.concat([all_df, original_encoded], axis=0)
    ball_df = __sort_dataframe_by_integer_index(ball_df)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/all_original_ball_l{lottery}.csv')
    ball_df.to_csv(path_to_dataframe)
    star_df = __sort_dataframe_by_integer_index(star_df)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/all_original_star_{lottery}.csv')
    star_df.to_csv(path_to_dataframe)
    all_df = __sort_dataframe_by_integer_index(all_df)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/all_original_{lottery}.csv')
    all_df.to_csv(path_to_dataframe)
    return {'all_df': all_df, 'ball_df': ball_df, 'star_df': star_df}

def concatenate_one_hot(df_one_hot, lottery):
    def concat_one_hot_variable(group):
        return np.sum(group)
    df_concat_all_df = df_one_hot['all_df'].copy()
    df_concat_all_df['date'] = [int(index[:8]) for index in df_concat_all_df.index]
    df_concat_all_df = df_concat_all_df.groupby('date').apply(concat_one_hot_variable)
    df_concat_all_df.drop(columns=['date'], inplace=True)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/concat_all_one_hot_ball_{lottery}.csv')
    df_concat_all_df.to_csv(path_to_dataframe)
    df_concat_ball_df = df_one_hot['star_df'].copy()
    df_concat_ball_df['date'] = [int(index[:8]) for index in df_concat_ball_df.index]
    df_concat_ball_df = df_concat_ball_df.groupby('date').apply(concat_one_hot_variable)
    df_concat_ball_df.drop(columns=['date'], inplace=True)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/concat_all_one_hot_star_{lottery}.csv')
    df_concat_ball_df.to_csv(path_to_dataframe)
    df_concat_star_df = df_one_hot['all_df'].copy()
    df_concat_star_df['date'] = [int(index[:8]) for index in df_concat_star_df.index]
    df_concat_star_df = df_concat_star_df.groupby('date').apply(concat_one_hot_variable)
    df_concat_star_df.drop(columns=['date'], inplace=True)
    path_to_dataframe = os.path.join(os.getcwd(), f'data/concat_all_one_hot_{lottery}.csv')
    df_concat_ball_df.to_csv(path_to_dataframe)
    return {'all_df': df_concat_all_df, 'ball_df': df_concat_ball_df, 'star_df': df_concat_star_df}   


def main(lottery, mode='concatenate_one_hot'):

    if lottery == 'loto':
        root_data = '/home/yanncauchepin/Datasets/Lottery/loto'
        path_to_dataframe = os.path.join('/home/yanncauchepin/Git/Lottery/data', 'all_loto.csv')
    elif lottery == 'euromillions':
        root_data = '/home/yanncauchepin/Datasets/Lottery/euromillions'
        path_to_dataframe = os.path.join('/home/yanncauchepin/Git/Lottery/data', 'all_euromillions.csv')
    else:
        raise ValueError()

    df = build_dataframe(lottery, root_data, path_to_dataframe)
    if mode == 'one_hot':
        return one_hot_encoding(df, lottery)
    elif mode == 'original':
        return original_encoding(df, lottery)
    elif mode == 'concatenate_one_hot':
        return concatenate_one_hot(one_hot_encoding(df, lottery), lottery)
    else:
        return ValueError()

if __name__=='__main__':
    print(main('euromillions', 'concatenate_one_hot'))
        