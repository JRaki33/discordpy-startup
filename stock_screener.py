"""
株式スクリーナー
日本株（東証）・米株（NYSE/NASDAQ）の低位株をスクリーニングして分析する
"""
from typing import Optional
from stock_analyzer import analyze_stock, StockAnalysis, format_analysis

# ---- 日本株デフォルト監視リスト ----
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
    "4208.T",  # UBE
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

# ---- 米株デフォルト監視リスト（低価格帯〜中価格帯の流動性の高い銘柄） ----
DEFAULT_US_WATCHLIST = [
    # テクノロジー
    "INTC",   # Intel
    "CSCO",   # Cisco
    "HPQ",    # HP Inc.
    "DELL",   # Dell Technologies
    "STX",    # Seagate Technology
    "WDC",    # Western Digital
    "SNAP",   # Snap
    "PINS",   # Pinterest
    "OPEN",   # Opendoor Technologies
    # 金融
    "BAC",    # Bank of America
    "WFC",    # Wells Fargo
    "C",      # Citigroup
    "USB",    # U.S. Bancorp
    "RF",     # Regions Financial
    "KEY",    # KeyCorp
    "FITB",   # Fifth Third Bancorp
    # エネルギー
    "XOM",    # ExxonMobil
    "CVX",    # Chevron
    "OXY",    # Occidental Petroleum
    "MRO",    # Marathon Oil
    "DVN",    # Devon Energy
    # 素材・工業
    "FCX",    # Freeport-McMoRan（銅）
    "CLF",    # Cleveland-Cliffs（鉄鋼）
    "X",      # US Steel
    "AA",     # Alcoa（アルミ）
    "NUE",    # Nucor
    # 消費財・小売
    "F",      # Ford Motor
    "GM",     # General Motors
    "M",      # Macy's
    "KHC",    # Kraft Heinz
    "PFE",    # Pfizer
]


def _screen(
    watchlist: list[str],
    max_price: float,
    min_rise_probability: float,
    top_n: int,
    weights: Optional[dict],
    reversal_only: bool,
) -> list[StockAnalysis]:
    """スクリーニング共通ロジック"""
    results = []
    for ticker in watchlist:
        analysis = analyze_stock(ticker, weights=weights)
        if analysis is None:
            continue
        if analysis.current_price > max_price:
            continue
        if reversal_only:
            if not analysis.reversal_detected:
                continue
        else:
            if analysis.rise_probability < min_rise_probability:
                continue
        results.append(analysis)
    results.sort(key=lambda x: x.rise_probability, reverse=True)
    return results[:top_n]


def screen_low_price_stocks(
    watchlist: list[str] = None,
    max_price: float = 1000.0,
    min_rise_probability: float = 60.0,
    top_n: int = 5,
    weights: Optional[dict] = None,
) -> list[StockAnalysis]:
    """日本株の低位株をスクリーニングして上昇確率上位を返す"""
    return _screen(
        watchlist or DEFAULT_WATCHLIST,
        max_price, min_rise_probability, top_n, weights, reversal_only=False,
    )


def screen_reversal_stocks(
    watchlist: list[str] = None,
    max_price: float = 1000.0,
    top_n: int = 5,
    weights: Optional[dict] = None,
) -> list[StockAnalysis]:
    """日本株の反転シグナル銘柄を返す"""
    return _screen(
        watchlist or DEFAULT_WATCHLIST,
        max_price, 0, top_n, weights, reversal_only=True,
    )


def screen_us_low_price_stocks(
    watchlist: list[str] = None,
    max_price: float = 50.0,
    min_rise_probability: float = 60.0,
    top_n: int = 5,
    weights: Optional[dict] = None,
) -> list[StockAnalysis]:
    """米株の低位株をスクリーニングして上昇確率上位を返す（デフォルト$50以下）"""
    return _screen(
        watchlist or DEFAULT_US_WATCHLIST,
        max_price, min_rise_probability, top_n, weights, reversal_only=False,
    )


def screen_us_reversal_stocks(
    watchlist: list[str] = None,
    max_price: float = 50.0,
    top_n: int = 5,
    weights: Optional[dict] = None,
) -> list[StockAnalysis]:
    """米株の反転シグナル銘柄を返す"""
    return _screen(
        watchlist or DEFAULT_US_WATCHLIST,
        max_price, 0, top_n, weights, reversal_only=True,
    )


def _format_stock_section(stocks: list[StockAnalysis], label: str) -> list[str]:
    lines = [f"## {label}"]
    if stocks:
        for i, s in enumerate(stocks, 1):
            price_str = f"${s.current_price:,.2f}" if s.currency == "USD" else f"¥{s.current_price:,.0f}"
            lines.append(f"\n### {i}位 {s.company_name} ({s.ticker})")
            lines.append(f"価格: {price_str}  上昇確率: **{s.rise_probability:.0f}%**")
            lines.append(s.summary)
    else:
        lines.append("条件を満たす銘柄が見つかりませんでした。")
    return lines


def _format_reversal_section(stocks: list[StockAnalysis], label: str) -> list[str]:
    lines = [f"## {label}"]
    if stocks:
        for s in stocks:
            price_str = f"${s.current_price:,.2f}" if s.currency == "USD" else f"¥{s.current_price:,.0f}"
            lines.append(
                f"\n**{s.company_name}** ({s.ticker})  {price_str}  上昇確率: {s.rise_probability:.0f}%"
            )
            lines.append(f"  → {s.summary}")
            if s.signals:
                lines.append(f"  主なシグナル: {s.signals[0]}")
    else:
        lines.append("現時点で反転シグナルの銘柄はありません。")
    return lines


def format_daily_report(
    jp_top: list[StockAnalysis],
    jp_reversal: list[StockAnalysis],
    us_top: list[StockAnalysis] = None,
    us_reversal: list[StockAnalysis] = None,
) -> str:
    """毎日の朝レポートをフォーマット（日本株＋米株）"""
    from datetime import datetime

    today = datetime.now().strftime("%Y年%m月%d日")
    lines = [f"# 📊 デイリーレポート [{today}]", ""]

    lines += ["---"] + _format_stock_section(jp_top, "🇯🇵 注目日本株 TOP（上昇確率順）")
    lines += ["", "---"] + _format_reversal_section(jp_reversal, "🇯🇵 日本株 反転シグナル検出銘柄")

    if us_top is not None:
        lines += ["", "---"] + _format_stock_section(us_top, "🇺🇸 注目米株 TOP（上昇確率順）")
    if us_reversal is not None:
        lines += ["", "---"] + _format_reversal_section(us_reversal, "🇺🇸 米株 反転シグナル検出銘柄")

    lines += ["", "---", "⚠️ *本レポートは自動分析です。投資の最終判断はご自身でお願いします。*"]
    return "\n".join(lines)
