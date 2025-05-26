import pandas as pd
from pandas.tseries.offsets import MonthEnd
import yfinance as yf
from typing import List, Optional


def load_jp_cpi_df() -> pd.DataFrame:
    """
    日本のCPIデータを読み込み、整形したDataFrameを返す。
    
    Returns:
        pd.DataFrame: 日付とCPIのカラムを持つDataFrame
    """
    df = pd.read_excel('data/jp_cpi.xlsx', header=7)[['Unnamed: 1', 'Unnamed: 12']].dropna() # https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00200573&tstat=000001150147&cycle=1&year=20250&month=12040604&tclass1=000001150149
    df.columns = ['Date', 'cpi']
    df['Date'] = pd.to_datetime(df['Date'].astype(int).astype(str), format='%Y%m') + MonthEnd(0)
    df = df.reset_index(drop=True)
    return df


def get_etf_close_prices(
        tickers: Optional[List[str]] = None,
        start_date: str = '2000-01-01',
        end_date: str = '2024-12-31'
    ) -> pd.DataFrame:
    """
    複数ETFの終値データを取得し、日付をインデックスとしたDataFrameを返す。
    
    Args:
        tickers: ETFティッカーのリスト。Noneの場合はデフォルトティッカーを使用
        start_date: 取得開始日（YYYY-MM-DD形式）
        end_date: 取得終了日（YYYY-MM-DD形式）
        
    Returns:
        pd.DataFrame: 日付カラムと各ETFの終値を列とするDataFrame。取得失敗時は空のDataFrame
    """
    if tickers is None:
        tickers = ['VTI', 'VEA', 'VWO', 'IAU', 'IYR']
    
    try:
        etf_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if etf_data is None or len(etf_data) == 0:
            return pd.DataFrame()
        
        # 単一ティッカーの場合とMultiIndexの場合を処理
        if len(tickers) == 1:
            result = etf_data[['Close']].rename(columns={'Close': tickers[0]})
            result = result.reset_index()  # 日付をカラムとして追加
            return result
        
        close_data = []
        for ticker in tickers:
            if ('Close', ticker) in etf_data.columns:
                series = etf_data[('Close', ticker)].dropna().rename(ticker)
                close_data.append(series)
        
        if not close_data:
            return pd.DataFrame()
            
        result = pd.concat(close_data, axis=1).dropna()
        result = result.reset_index()  # 日付をカラムとして追加
        return result
        
    except Exception as e:
        print(f"ETFデータ取得エラー: {e}")
        return pd.DataFrame()


def get_usd_jpy_rate(
        start_date: str = '2000-01-01',
        end_date: str = '2024-12-31'
    ) -> pd.DataFrame:
    """
    ドル円為替レートを取得し、日付をカラムとしたDataFrameを返す。
    
    Args:
        start_date: 取得開始日（YYYY-MM-DD形式）
        end_date: 取得終了日（YYYY-MM-DD形式）
        
    Returns:
        pd.DataFrame: 日付とドル円為替レートのカラムを持つDataFrame。取得失敗時は空のDataFrame
    """
    try:
        # ドル円為替レートを取得
        usd_jpy_data = yf.download('USDJPY=X', start=start_date, end=end_date, progress=False)
        
        if usd_jpy_data is None or len(usd_jpy_data) == 0:
            return pd.DataFrame()
        
        # MultiIndexの場合は通常のカラムに変換
        if isinstance(usd_jpy_data.columns, pd.MultiIndex):
            usd_jpy_data.columns = usd_jpy_data.columns.droplevel(1)
        
        # Closeカラムのみを取得し、カラム名を変更
        result = usd_jpy_data[['Close']].rename(columns={'Close': 'USD_JPY'})
        result = result.reset_index()  # 日付をカラムとして追加
        
        # カラム名を削除
        result.columns.name = None
        
        return result
        
    except Exception as e:
        print(f"ドル円為替レート取得エラー: {e}")
        return pd.DataFrame()