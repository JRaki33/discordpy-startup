"""
低位株スクリーナー
東証上場の低位株（株価500円未満など）をスクリーニングして分析する
"""
from typing import Optional
from stock_analyzer import analyze_stock, StockAnalysis, format_analysis

# ---- 低位株として監視するデフォルト銘柄リスト ----
# 株価が比較的低い東証銘柄を初期値として設定（ユーザーが自由に変更可）
# yfinance形式: コード + ".T"
DEFAULT_WATCHLIST = [
    "1301.T",  # 極洋
    "1332.T",  # 日本水産
    "1333.T",  # マルハニチロ
    "2002.T",  # 日清製粉G
    "2809.T",  # キユーピー
    "3436.T",  # SUMCO
    "4004.T",  # レゾナック
    "4061.T",  # デンカ
    "4202.T",  # ダイセル
    "4208.T",  # 宇部興産→UBE
    "4217.T",  # 日立化成→昭和電工M
    "4401.T",  # ADEKA
    "5020.T",  # ENEOSホールディングス
    "5201.T",  # AGC
    "5233.T",  # 太平洋セメント
    "5301.T",  # 東海カーボン
    "5401.T",  # 日本製鉄
    "5411.T",  # JFEホールディングス
    "5631.T",  # 日本製鋼所
    "6301.T",  # 小松製作所
    "7011.T",  # 三菱重工業
    "7201.T",  # 日産自動車
    "7203.T",  # トヨタ自動車
    "7267.T",  # ホンダ
    "8001.T",  # 伊藤忠商事
    "8306.T",  # 三菱UFJ FG
    "8316.T",  # 三井住友FG
    "9101.T",  # 日本郵船
    "9104.T",  # 商船三井
    "9107.T",  # 川崎汽船
]


def screen_low_price_stocks(
    watchlist: list[str] = None,
    max_price: float = 1000.0,
    min_rise_probability: float = 60.0,
    top_n: int = 5,
    weights: Optional[dict] = None,
) -> list[StockAnalysis]:
    """
    低位株をスクリーニングして上昇確率の高い銘柄を返す

    Args:
        watchlist: 対象ティッカーリスト (None でデフォルト)
        max_price: この価格以下の銘柄だけ対象 (低位株フィルター)
        min_rise_probability: 上昇確率の足切りライン
        top_n: 返す件数
        weights: シグナル重み辞書（prediction_db から取得）
    """
    if watchlist is None:
        watchlist = DEFAULT_WATCHLIST

    results = []
    for ticker in watchlist:
        analysis = analyze_stock(ticker, weights=weights)
        if analysis is None:
            continue
        if analysis.current_price > max_price:
            continue
        if analysis.rise_probability < min_rise_probability:
            continue
        results.append(analysis)

    results.sort(key=lambda x: x.rise_probability, reverse=True)
    return results[:top_n]


def screen_reversal_stocks(
    watchlist: list[str] = None,
    max_price: float = 1000.0,
    top_n: int = 5,
    weights: Optional[dict] = None,
) -> list[StockAnalysis]:
    """
    反転シグナルが出ている低位株を返す

    Args:
        watchlist: 対象ティッカーリスト
        max_price: 低位株フィルター
        top_n: 返す件数
        weights: シグナル重み辞書
    """
    if watchlist is None:
        watchlist = DEFAULT_WATCHLIST

    results = []
    for ticker in watchlist:
        analysis = analyze_stock(ticker, weights=weights)
        if analysis is None:
            continue
        if analysis.current_price > max_price:
            continue
        if not analysis.reversal_detected:
            continue
        results.append(analysis)

    results.sort(key=lambda x: x.rise_probability, reverse=True)
    return results[:top_n]


def format_daily_report(
    top_stocks: list[StockAnalysis],
    reversal_stocks: list[StockAnalysis],
) -> str:
    """毎日の朝レポートをフォーマット"""
    from datetime import datetime

    today = datetime.now().strftime("%Y年%m月%d日")
    lines = [
        f"# 📊 低位株デイリーレポート [{today}]",
        "",
        "---",
        "## 🚀 本日の注目低位株 TOP（上昇確率順）",
    ]

    if top_stocks:
        for i, stock in enumerate(top_stocks, 1):
            lines.append(f"\n### {i}位 {stock.company_name} ({stock.ticker})")
            lines.append(
                f"価格: ¥{stock.current_price:,.0f}  "
                f"上昇確率: **{stock.rise_probability:.0f}%**"
            )
            lines.append(stock.summary)
    else:
        lines.append("条件を満たす銘柄が見つかりませんでした。")

    lines += [
        "",
        "---",
        "## 🔄 反転シグナル検出銘柄",
    ]

    if reversal_stocks:
        for stock in reversal_stocks:
            lines.append(
                f"\n**{stock.company_name}** ({stock.ticker})  "
                f"¥{stock.current_price:,.0f}  "
                f"上昇確率: {stock.rise_probability:.0f}%"
            )
            lines.append(f"  → {stock.summary}")
            if stock.signals:
                lines.append(f"  主なシグナル: {stock.signals[0]}")
    else:
        lines.append("現時点で反転シグナルの銘柄はありません。")

    lines += [
        "",
        "---",
        "⚠️ *本レポートは自動分析です。投資の最終判断はご自身でお願いします。*",
    ]

    return "\n".join(lines)
