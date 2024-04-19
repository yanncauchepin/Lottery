import os
import pandas as pd

root_data = '/media/yanncauchepin/ExternalDisk/Datasets/Lottery/euromillions'

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

if __name__=='__main__':
    data = {}
    for file_dataframe in os.listdir(root_data):
        print(file_dataframe)
        file_dataframe = os.path.join(root_data, file_dataframe)
        df = pd.read_csv(file_dataframe, **csv_options)
        dict_info = extract_informations(df)
        data.update(dict_info)
    dataframe = pd.DataFrame(data=data)
    dataframe = dataframe.sort_index(axis=1, ascending=True)
    path_to_dataframe = os.path.join(os.getcwd(), 'all_euromillions.csv')
    dataframe.to_csv(path_to_dataframe)
    
    