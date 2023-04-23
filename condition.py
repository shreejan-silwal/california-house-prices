import pandas as pd

def get_file(file_name):
    df = pd.read_csv(file_name)

    ocean_proximity_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity').astype(int)
    df = pd.concat([df.drop(['median_house_value', 'ocean_proximity' ], axis=1), ocean_proximity_dummies, df['median_house_value']], axis=1)
    df = df.dropna()
    return df


file1 = get_file('./training_data.csv')
file2 = get_file('./test_data.csv')

file1.to_csv('./train_conditioned.csv', index = False)
file2.to_csv('./test_conditioned.csv', index = False)
