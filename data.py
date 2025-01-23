import pandas as pd
from strlearn.streams import CSVParser

def convert_label(label: str) -> int :
    if label == 'positive':
        return 1
    else:
        return -1

def get_dataframe(n : int = 10000, weights : tuple[float,float] = (0.5,0.5)) -> pd.DataFrame:
    if weights[0] > 1 or weights[1] > 1 or weights[0] < 0 or weights[1] < 0:
        raise ValueError('invalid weight value: ', weights)
    
    rows = int(n/2)
    df = pd.read_csv('dataset.csv')
    df['sentiment'] = df['sentiment'].apply(convert_label)
    return df.groupby('sentiment').head(rows)