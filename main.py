import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from utils import load_jp_cpi_df, get_etf_close_prices, get_usd_jpy_rate


def prepare_data():
    """
    各データを取得し、共通の期間で統合する
    """
    print("データを取得中...")
    
    # データ取得
    cpi_df = load_jp_cpi_df()
    etf_df = get_etf_close_prices(tickers=['VTI', 'VEA', 'VWO', 'IAU', 'IYR'])
    usd_jpy = get_usd_jpy_rate()
    
    print(f"CPI データ期間: {cpi_df['Date'].min()} - {cpi_df['Date'].max()}")
    print(f"ETF データ期間: {etf_df['Date'].min()} - {etf_df['Date'].max()}")
    print(f"USD/JPY データ期間: {usd_jpy['Date'].min()} - {usd_jpy['Date'].max()}")
    
    # ETFデータを円建てに変換
    etf_jpy = etf_df.copy()
    
    # 為替レートをマージ
    etf_jpy = pd.merge(etf_jpy, usd_jpy, on='Date', how='inner')
    
    # ETF価格を円建てに変換
    for col in ['VTI', 'VEA', 'VWO', 'IAU', 'IYR']:
        if col in etf_jpy.columns:
            etf_jpy[col] = etf_jpy[col] * etf_jpy['USD_JPY']
    
    # USD_JPY列を削除
    etf_jpy = etf_jpy.drop('USD_JPY', axis=1)
    
    # 月次リターン計算
    etf_jpy['Date'] = pd.to_datetime(etf_jpy['Date'])
    etf_jpy = etf_jpy.set_index('Date')
    monthly_etf = etf_jpy.resample('M').last()
    etf_returns = monthly_etf.pct_change().dropna()
    
    # CPIデータの月次インフレ率計算
    cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])
    cpi_df = cpi_df.set_index('Date')
    inflation_rate = cpi_df['cpi'].pct_change(12).dropna()  # 年率インフレ率
    
    # 共通期間でのデータ統合
    common_dates = etf_returns.index.intersection(inflation_rate.index)
    etf_returns_aligned = etf_returns.loc[common_dates]
    inflation_aligned = inflation_rate.loc[common_dates]
    
    print(f"共通データ期間: {common_dates.min()} - {common_dates.max()}")
    print(f"共通データポイント数: {len(common_dates)}")
    
    return etf_returns_aligned, inflation_aligned


def portfolio_performance(weights, returns, inflation_rate):
    """
    ポートフォリオのパフォーマンスを計算
    """
    portfolio_return = np.sum(returns.mean() * weights) * 12  # 年率リターン
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 12, weights)))  # 年率標準偏差
    avg_inflation = inflation_rate.mean()
    excess_return = portfolio_return - avg_inflation
    
    # 対インフレ超過リターンの標準偏差を計算
    portfolio_monthly_returns = (returns * weights).sum(axis=1)
    monthly_inflation = inflation_rate / 12  # 月次インフレ率
    excess_returns_series = portfolio_monthly_returns - monthly_inflation
    excess_return_std = excess_returns_series.std() * np.sqrt(12)  # 年率化
    
    # シャープレシオを対インフレ超過リターンの標準偏差で計算
    sharpe_ratio = excess_return / excess_return_std if excess_return_std > 0 else 0
    
    return portfolio_return, portfolio_std, excess_return, sharpe_ratio


def calculate_rolling_sharpe_ratios(weights, returns, inflation_rate, window_years=15):
    """
    任意の15年間における対インフレシャープレシオを計算
    """
    window_months = window_years * 12
    portfolio_monthly_returns = (returns * weights).sum(axis=1)
    monthly_inflation = inflation_rate / 12
    
    sharpe_ratios = []
    
    for i in range(len(portfolio_monthly_returns) - window_months + 1):
        # 15年間のウィンドウでデータを取得
        window_portfolio_returns = portfolio_monthly_returns.iloc[i:i+window_months]
        window_inflation = monthly_inflation.iloc[i:i+window_months]
        
        # 超過リターンを計算
        excess_returns = window_portfolio_returns - window_inflation
        
        # 平均超過リターンと標準偏差を計算（年率化）
        mean_excess_return = excess_returns.mean() * 12
        std_excess_return = excess_returns.std() * np.sqrt(12)
        
        # シャープレシオを計算
        if std_excess_return > 0:
            sharpe_ratio = mean_excess_return / std_excess_return
            sharpe_ratios.append(sharpe_ratio)
    
    return np.array(sharpe_ratios)


def objective_function_min_sharpe(weights, returns, inflation_rate):
    """
    最適化の目的関数（15年間の対インフレシャープレシオの最小値を最大化）
    """
    sharpe_ratios = calculate_rolling_sharpe_ratios(weights, returns, inflation_rate)
    
    if len(sharpe_ratios) == 0:
        return float('inf')  # データが不足している場合は大きなペナルティ
    
    min_sharpe = np.min(sharpe_ratios)
    # 最小値を最大化（負の値を返して最小化問題に変換）
    return -min_sharpe


def objective_function_mean_sharpe(weights, returns, inflation_rate):
    """
    最適化の目的関数（15年間の対インフレシャープレシオの平均値を最大化）
    """
    sharpe_ratios = calculate_rolling_sharpe_ratios(weights, returns, inflation_rate)
    
    if len(sharpe_ratios) == 0:
        return float('inf')  # データが不足している場合は大きなペナルティ
    
    mean_sharpe = np.mean(sharpe_ratios)
    # 平均値を最大化（負の値を返して最小化問題に変換）
    return -mean_sharpe


def find_optimal_portfolio(returns, inflation_rate, objective_type='min_sharpe'):
    """
    最適なポートフォリオ比率を求める
    
    Parameters:
    - objective_type: 'min_sharpe' (最小値最大化) または 'mean_sharpe' (平均値最大化)
    """
    n_assets = len(returns.columns)
    
    # 制約条件
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 重みの合計が1
    bounds = tuple([(0, 1) for _ in range(n_assets)])  # 各重みは0-1の範囲
    
    # 初期値（等ウェイトポートフォリオ）
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # 目的関数を選択
    if objective_type == 'min_sharpe':
        objective_func = objective_function_min_sharpe
    elif objective_type == 'mean_sharpe':
        objective_func = objective_function_mean_sharpe
    else:
        raise ValueError("objective_type must be 'min_sharpe' or 'mean_sharpe'")
    
    # 最適化実行
    result = minimize(objective_func, initial_weights, 
                     args=(returns, inflation_rate),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result


def analyze_rolling_performance(weights, returns, inflation_rate):
    """
    15年間のローリングパフォーマンスを分析
    """
    sharpe_ratios = calculate_rolling_sharpe_ratios(weights, returns, inflation_rate)
    
    if len(sharpe_ratios) == 0:
        return None
    
    analysis = {
        'median_sharpe': np.median(sharpe_ratios),
        'mean_sharpe': np.mean(sharpe_ratios),
        'min_sharpe': np.min(sharpe_ratios),
        'max_sharpe': np.max(sharpe_ratios),
        'std_sharpe': np.std(sharpe_ratios),
        'positive_periods': np.sum(sharpe_ratios > 0) / len(sharpe_ratios)
    }
    
    return analysis


def analyze_portfolio_scenarios(etf_returns, inflation_rate):
    """
    異なる投資戦略の比較分析
    """
    tickers = ['VTI', 'VEA', 'VWO', 'IAU', 'IYR']
    
    # 最小値最大化ポートフォリオ
    min_optimal_result = find_optimal_portfolio(etf_returns, inflation_rate, 'min_sharpe')
    if not min_optimal_result.success:
        print("最小値最大化の最適化に失敗しました:", min_optimal_result.message)
        return
    
    # 平均値最大化ポートフォリオ
    mean_optimal_result = find_optimal_portfolio(etf_returns, inflation_rate, 'mean_sharpe')
    if not mean_optimal_result.success:
        print("平均値最大化の最適化に失敗しました:", mean_optimal_result.message)
        return
    
    min_optimal_weights = min_optimal_result.x
    mean_optimal_weights = mean_optimal_result.x
    
    # パフォーマンス計算
    min_opt_return, min_opt_std, min_opt_excess, min_opt_sharpe = portfolio_performance(min_optimal_weights, etf_returns, inflation_rate)
    mean_opt_return, mean_opt_std, mean_opt_excess, mean_opt_sharpe = portfolio_performance(mean_optimal_weights, etf_returns, inflation_rate)
    
    # 等重みポートフォリオ
    equal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    eq_return, eq_std, eq_excess, eq_sharpe = portfolio_performance(equal_weights, etf_returns, inflation_rate)
    
    # 結果表示
    print("\n" + "="*80)
    print("ポートフォリオ分析結果")
    print("="*80)
    
    print("\n■ 最小値最大化ポートフォリオの投資比率:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {min_optimal_weights[i]:6.1%}")
    
    print("\n■ 平均値最大化ポートフォリオの投資比率:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {mean_optimal_weights[i]:6.1%}")
    
    avg_inflation = inflation_rate.mean()
    print(f"\n■ 最小値最大化ポートフォリオのパフォーマンス指標:")
    print(f"  ポートフォリオ年率リターン: {min_opt_return:6.2%}")
    print(f"  平均インフレ率:           {avg_inflation:6.2%}")
    print(f"  インフレ超過リターン:     {min_opt_excess:6.2%}")
    print(f"  ポートフォリオリスク:     {min_opt_std:6.2%}")
    print(f"  シャープレシオ:           {min_opt_sharpe:6.3f}")
    
    print(f"\n■ 平均値最大化ポートフォリオのパフォーマンス指標:")
    print(f"  ポートフォリオ年率リターン: {mean_opt_return:6.2%}")
    print(f"  平均インフレ率:           {avg_inflation:6.2%}")
    print(f"  インフレ超過リターン:     {mean_opt_excess:6.2%}")
    print(f"  ポートフォリオリスク:     {mean_opt_std:6.2%}")
    print(f"  シャープレシオ:           {mean_opt_sharpe:6.3f}")
    
    # 15年間のローリングパフォーマンス分析
    min_rolling_analysis = analyze_rolling_performance(min_optimal_weights, etf_returns, inflation_rate)
    mean_rolling_analysis = analyze_rolling_performance(mean_optimal_weights, etf_returns, inflation_rate)
    eq_rolling_analysis = analyze_rolling_performance(equal_weights, etf_returns, inflation_rate)
    
    if min_rolling_analysis:
        print(f"\n■ 最小値最大化ポートフォリオの15年間ローリング対インフレシャープレシオ分析:")
        print(f"  中央値:                   {min_rolling_analysis['median_sharpe']:6.3f}")
        print(f"  平均値:                   {min_rolling_analysis['mean_sharpe']:6.3f}")
        print(f"  最小値:                   {min_rolling_analysis['min_sharpe']:6.3f}  ← 最適化対象")
        print(f"  最大値:                   {min_rolling_analysis['max_sharpe']:6.3f}")
        print(f"  標準偏差:                 {min_rolling_analysis['std_sharpe']:6.3f}")
        print(f"  正の期間割合:             {min_rolling_analysis['positive_periods']:6.1%}")
    
    if mean_rolling_analysis:
        print(f"\n■ 平均値最大化ポートフォリオの15年間ローリング対インフレシャープレシオ分析:")
        print(f"  中央値:                   {mean_rolling_analysis['median_sharpe']:6.3f}")
        print(f"  平均値:                   {mean_rolling_analysis['mean_sharpe']:6.3f}  ← 最適化対象")
        print(f"  最小値:                   {mean_rolling_analysis['min_sharpe']:6.3f}")
        print(f"  最大値:                   {mean_rolling_analysis['max_sharpe']:6.3f}")
        print(f"  標準偏差:                 {mean_rolling_analysis['std_sharpe']:6.3f}")
        print(f"  正の期間割合:             {mean_rolling_analysis['positive_periods']:6.1%}")
    
    print(f"\n■ 15年間ローリングシャープレシオの比較:")
    print(f"{'戦略':25} {'平均値':>8} {'最小値':>8} {'最大値':>8} {'標準偏差':>8}")
    print("-" * 65)
    if min_rolling_analysis:
        print(f"{'最小値最大化ポートフォリオ':25} {min_rolling_analysis['mean_sharpe']:7.3f} {min_rolling_analysis['min_sharpe']:7.3f} {min_rolling_analysis['max_sharpe']:7.3f} {min_rolling_analysis['std_sharpe']:7.3f}")
    if mean_rolling_analysis:
        print(f"{'平均値最大化ポートフォリオ':25} {mean_rolling_analysis['mean_sharpe']:7.3f} {mean_rolling_analysis['min_sharpe']:7.3f} {mean_rolling_analysis['max_sharpe']:7.3f} {mean_rolling_analysis['std_sharpe']:7.3f}")
    if eq_rolling_analysis:
        print(f"{'等重みポートフォリオ':25} {eq_rolling_analysis['mean_sharpe']:7.3f} {eq_rolling_analysis['min_sharpe']:7.3f} {eq_rolling_analysis['max_sharpe']:7.3f} {eq_rolling_analysis['std_sharpe']:7.3f}")
    
    # シナリオ比較表
    print(f"\n■ 投資戦略の比較:")
    print(f"{'戦略':25} {'年率リターン':>10} {'超過リターン':>10} {'リスク':>8} {'シャープ':>8}")
    print("-" * 75)
    print(f"{'最小値最大化ポートフォリオ':25} {min_opt_return:9.2%} {min_opt_excess:9.2%} {min_opt_std:7.2%} {min_opt_sharpe:7.3f}")
    print(f"{'平均値最大化ポートフォリオ':25} {mean_opt_return:9.2%} {mean_opt_excess:9.2%} {mean_opt_std:7.2%} {mean_opt_sharpe:7.3f}")
    print(f"{'等重みポートフォリオ':25} {eq_return:9.2%} {eq_excess:9.2%} {eq_std:7.2%} {eq_sharpe:7.3f}")
    
    # リスク分析（最小値最大化ポートフォリオ）
    min_portfolio_returns = (etf_returns * min_optimal_weights).sum(axis=1)
    min_cumulative_returns = (1 + min_portfolio_returns).cumprod()
    min_rolling_max = min_cumulative_returns.expanding().max()
    min_drawdowns = (min_cumulative_returns - min_rolling_max) / min_rolling_max
    min_max_drawdown = min_drawdowns.min()
    
    min_var_95 = np.percentile(min_portfolio_returns, 5)
    monthly_inflation = inflation_rate / 12
    min_underperform_probability = np.mean(min_portfolio_returns < monthly_inflation)
    
    # リスク分析（平均値最大化ポートフォリオ）
    mean_portfolio_returns = (etf_returns * mean_optimal_weights).sum(axis=1)
    mean_cumulative_returns = (1 + mean_portfolio_returns).cumprod()
    mean_rolling_max = mean_cumulative_returns.expanding().max()
    mean_drawdowns = (mean_cumulative_returns - mean_rolling_max) / mean_rolling_max
    mean_max_drawdown = mean_drawdowns.min()
    
    mean_var_95 = np.percentile(mean_portfolio_returns, 5)
    mean_underperform_probability = np.mean(mean_portfolio_returns < monthly_inflation)
    
    print(f"\n■ リスク指標比較:")
    print(f"{'戦略':25} {'最大DD':>8} {'VaR(95%)':>9} {'ｲﾝﾌﾚ下回確率':>12}")
    print("-" * 60)
    print(f"{'最小値最大化ポートフォリオ':25} {min_max_drawdown:7.2%} {min_var_95:8.2%} {min_underperform_probability:11.1%}")
    print(f"{'平均値最大化ポートフォリオ':25} {mean_max_drawdown:7.2%} {mean_var_95:8.2%} {mean_underperform_probability:11.1%}")
    
    print(f"\n■ 投資実行時の注意点:")
    print(f"  - 最小値最大化: 悪いシナリオでの下振れリスクを重視")
    print(f"  - 平均値最大化: 長期的な平均的なパフォーマンスを重視")
    print(f"  - 定期的なリバランシングが必要")
    print(f"  - 為替リスクを考慮したヘッジを検討")
    print(f"  - 市場状況に応じた投資比率の再評価")
    print(f"  - 長期投資の視点を維持")


def main():
    """
    メイン実行関数
    """
    try:
        # データ準備
        etf_returns, inflation_rate = prepare_data()
        
        # ポートフォリオ分析実行
        analyze_portfolio_scenarios(etf_returns, inflation_rate)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

