import os
import pandas as pd

root_data = '/home/yanncauchepin/Datasets/Lottery/euromillions'

csv_options = {
    'sep': ';',
    'quotechar': '"',
    'quoting': 0,
    'escapechar': None,
    'encoding': 'latin1',
    'index_col': False
    }

def extract_informations(original_dataframe):
    data = {}
    for i, draw in original_dataframe.iterrows():
        number = int(draw['annee_numero_de_tirage'])
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

def build_dataframe():
    data = {}
    for file_dataframe in os.listdir(root_data):
        print(file_dataframe)
        file_dataframe = os.path.join(root_data, file_dataframe)
        df = pd.read_csv(file_dataframe, **csv_options)
        dict_info = extract_informations(df)
        data.update(dict_info)
    dataframe = pd.DataFrame(data=data)
    dataframe = dataframe.sort_index(axis=1, ascending=True)
    path_to_dataframe = os.path.join(os.getcwd(), 'data/all_euromillions.csv')
    dataframe = dataframe.T
    dataframe.to_csv(path_to_dataframe)
    return dataframe

def __sort_dataframe_by_integer_index(df):
    integer_values = [int(index.split('__')[0]) for index in df.index]
    sorted_indices = [index for _, index in sorted(zip(integer_values, df.index))]
    sorted_df = df.loc[sorted_indices]
    return sorted_df

def one_hot_encoding(df):
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
    path_to_dataframe = os.path.join(os.getcwd(), 'data/all_one_hot_ball_euromillions.csv')
    ball_df.to_csv(path_to_dataframe)
    star_df = __sort_dataframe_by_integer_index(star_df)
    path_to_dataframe = os.path.join(os.getcwd(), 'data/all_one_hot_star_euromillions.csv')
    star_df.to_csv(path_to_dataframe)
    all_df = __sort_dataframe_by_integer_index(all_df)
    path_to_dataframe = os.path.join(os.getcwd(), 'data/all_one_hot_euromillions.csv')
    all_df.to_csv(path_to_dataframe)
    return {'all_df': all_df, 'ball_df': ball_df, 'star_df': star_df}

def original_encoding(df):
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
    path_to_dataframe = os.path.join(os.getcwd(), 'data/all_one_hot_ball_euromillions.csv')
    ball_df.to_csv(path_to_dataframe)
    star_df = __sort_dataframe_by_integer_index(star_df)
    path_to_dataframe = os.path.join(os.getcwd(), 'data/all_one_hot_star_euromillions.csv')
    star_df.to_csv(path_to_dataframe)
    all_df = __sort_dataframe_by_integer_index(all_df)
    path_to_dataframe = os.path.join(os.getcwd(), 'data/all_one_hot_euromillions.csv')
    all_df.to_csv(path_to_dataframe)
    return {'all_df': all_df, 'ball_df': ball_df, 'star_df': star_df}

def main(one_hot=True):
    df = build_dataframe()
    if one_hot:
        df = one_hot_encoding(df)
    else:
        df = original_encoding(df)
    
    return df 

if __name__=='__main__':
    main()
        