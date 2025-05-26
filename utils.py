import pandas as pd
from pandas.tseries.offsets import MonthEnd


def load_jp_cpi_df():
    df = pd.read_excel('data/jp_cpi.xlsx', header=7)[['Unnamed: 1', 'Unnamed: 12']].dropna() # https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00200573&tstat=000001150147&cycle=1&year=20250&month=12040604&tclass1=000001150149
    df.columns = ['date', 'cpi']
    df['date'] = pd.to_datetime(df['date'].astype(int).astype(str), format='%Y%m') + MonthEnd(0)
    df = df.reset_index(drop=True)
    return df