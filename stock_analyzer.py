"""
低位株分析モジュール
テクニカル指標を使って上昇確率・反転シグナルを計算する。
シグナルごとに動的重み（prediction_db から取得）を適用できる。
"""
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# シグナルキー定義: key -> (ベーススコア, 方向, 表示テキスト)
# 方向: "bull"=強気加点 / "bear"=弱気加点
SIGNAL_DEFS: dict[str, tuple[int, str, str]] = {
    "rsi_extreme_oversold":  (30, "bull", "🔴 RSI極端に売られすぎ(<25) → 反転期待"),
    "rsi_oversold":          (20, "bull", "🟠 RSI売られすぎ(<30) → 買いシグナル"),
    "rsi_overbought":        (10, "bear", "🔵 RSI高水準(>70) → 過熱注意"),
    "rsi_extreme_overbought":(20, "bear", "🔵 RSI買われすぎ(>75) → 反落注意"),
    "macd_golden_zero":      (25, "bull", "✅ MACDゴールデンクロス(ゼロライン下) → 強い買いシグナル"),
    "macd_histogram_rising": (15, "bull", "✅ MACDヒストグラム上昇中 → 上昇継続"),
    "macd_dead_cross":       (20, "bear", "❌ MACDデッドクロス → 売りシグナル"),
    "macd_falling":          (10, "bear", "⚠️ MACD下降中"),
    "bb_lower":              (20, "bull", "✅ ボリンジャーバンド下限タッチ → 反発期待"),
    "bb_upper":              (15, "bear", "⚠️ ボリンジャーバンド上限 → 反落注意"),
    "ma_golden_cross":       (20, "bull", "✅ 5日線が25日線をゴールデンクロス"),
    "ma_dead_cross":         (20, "bear", "❌ 5日線が25日線をデッドクロス"),
    "perfect_order_up":      (15, "bull", "📈 移動平均パーフェクトオーダー(上昇)"),
    "perfect_order_down":    (15, "bear", "📉 移動平均逆パーフェクトオーダー(下降)"),
    "volume_surge_up":       (15, "bull", "🔥 出来高急増+上昇 → 強い買い圧力"),
    "volume_surge_down":     (10, "bear", "⚠️ 出来高急増+下落 → 売り圧力注意"),
    "reversal_after_3down":  (15, "bull", "🔄 3日連続下落後の反転 → 底打ちシグナル"),
}


def is_us_stock(ticker: str) -> bool:
    """`.T` で終わらないティッカーを米株と判定"""
    return not ticker.upper().endswith(".T")


@dataclass
class StockAnalysis:
    ticker: str
    company_name: str
    current_price: float
    change_pct: float
    rise_probability: float       # 0〜100%
    signals: list[str]            # 表示用シグナルテキスト一覧
    signal_keys: list[str]        # DB記録用キー一覧
    reversal_detected: bool
    summary: str
    rsi: float
    macd_signal: str              # "bullish" / "bearish" / "neutral"
    bb_position: str              # "lower" / "middle" / "upper"
    trend: str                    # "uptrend" / "downtrend" / "sideways"
    volume_surge: bool
    market: str = "JP"            # "JP" or "US"
    currency: str = "JPY"         # "JPY" or "USD"


def fetch_stock_data(ticker: str, period_days: int = 120) -> Optional[pd.DataFrame]:
    """yfinanceで株価データを取得"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{period_days}d")
        if df.empty or len(df) < 20:
            return None
        return df
    except Exception:
        return None


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_bollinger(close: pd.Series, period: int = 20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + 2 * std, sma, sma - 2 * std


def _apply_signal(
    key: str,
    fired_keys: list[str],
    fired_texts: list[str],
    weights: dict[str, float],
) -> float:
    """シグナルを発火させてスコア変化量を返す"""
    base_score, direction, text = SIGNAL_DEFS[key]
    w = weights.get(key, 1.0)
    score = base_score * w
    fired_keys.append(key)
    fired_texts.append(text)
    return score if direction == "bull" else -score


def analyze_stock(
    ticker: str,
    weights: Optional[dict[str, float]] = None,
) -> Optional["StockAnalysis"]:
    """
    銘柄を分析して StockAnalysis を返す。

    Args:
        ticker:  yfinance 形式 (例: "9984.T")
        weights: シグナルキー→重み の辞書。None の場合は全て 1.0
    """
    if weights is None:
        weights = {}

    market = "US" if is_us_stock(ticker) else "JP"
    currency = "USD" if market == "US" else "JPY"

    df = fetch_stock_data(ticker)
    if df is None:
        return None

    close = df["Close"]
    volume = df["Volume"]

    rsi_series = calc_rsi(close)
    rsi = float(rsi_series.iloc[-1])
    macd, macd_sig, macd_hist = calc_macd(close)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)

    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    change_pct = (current_price - prev_price) / prev_price * 100

    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()

    vol_mean = volume.rolling(20).mean().iloc[-1]
    volume_surge = float(volume.iloc[-1]) >= vol_mean * 2.0

    fired_keys: list[str] = []
    fired_texts: list[str] = []
    net_score = 0.0

    # --- RSI ---
    if rsi < 25:
        net_score += _apply_signal("rsi_extreme_oversold", fired_keys, fired_texts, weights)
    elif rsi < 30:
        net_score += _apply_signal("rsi_oversold", fired_keys, fired_texts, weights)
    elif rsi > 75:
        net_score += _apply_signal("rsi_extreme_overbought", fired_keys, fired_texts, weights)
    elif rsi > 70:
        net_score += _apply_signal("rsi_overbought", fired_keys, fired_texts, weights)

    # --- MACD ---
    hist_now = float(macd_hist.iloc[-1])
    hist_prev = float(macd_hist.iloc[-2])
    macd_val = float(macd.iloc[-1])

    if macd_val < 0 and hist_now > hist_prev and hist_now > 0:
        net_score += _apply_signal("macd_golden_zero", fired_keys, fired_texts, weights)
        macd_signal = "bullish"
    elif macd_val > 0 and hist_now > hist_prev:
        net_score += _apply_signal("macd_histogram_rising", fired_keys, fired_texts, weights)
        macd_signal = "bullish"
    elif macd_val > 0 and hist_now < 0 and hist_prev >= 0:
        net_score += _apply_signal("macd_dead_cross", fired_keys, fired_texts, weights)
        macd_signal = "bearish"
    elif hist_now < hist_prev and hist_now < 0:
        net_score += _apply_signal("macd_falling", fired_keys, fired_texts, weights)
        macd_signal = "bearish"
    else:
        macd_signal = "neutral"

    # --- ボリンジャーバンド ---
    bb_upper_val = float(bb_upper.iloc[-1])
    bb_lower_val = float(bb_lower.iloc[-1])

    if current_price <= bb_lower_val:
        net_score += _apply_signal("bb_lower", fired_keys, fired_texts, weights)
        bb_position = "lower"
    elif current_price >= bb_upper_val:
        net_score += _apply_signal("bb_upper", fired_keys, fired_texts, weights)
        bb_position = "upper"
    else:
        bb_position = "middle"

    # --- 移動平均線 ---
    ma5_val = float(ma5.iloc[-1])
    ma25_val = float(ma25.iloc[-1])
    ma75_val = float(ma75.iloc[-1])
    ma5_prev = float(ma5.iloc[-2])
    ma25_prev = float(ma25.iloc[-2])

    if ma5_val > ma25_val and ma5_prev <= ma25_prev:
        net_score += _apply_signal("ma_golden_cross", fired_keys, fired_texts, weights)
    elif ma5_val < ma25_val and ma5_prev >= ma25_prev:
        net_score += _apply_signal("ma_dead_cross", fired_keys, fired_texts, weights)

    if current_price > ma25_val and ma25_val > ma75_val:
        net_score += _apply_signal("perfect_order_up", fired_keys, fired_texts, weights)
        trend = "uptrend"
    elif current_price < ma25_val and ma25_val < ma75_val:
        net_score += _apply_signal("perfect_order_down", fired_keys, fired_texts, weights)
        trend = "downtrend"
    else:
        trend = "sideways"

    # --- 出来高急増 ---
    if volume_surge:
        if change_pct > 0:
            net_score += _apply_signal("volume_surge_up", fired_keys, fired_texts, weights)
        else:
            net_score += _apply_signal("volume_surge_down", fired_keys, fired_texts, weights)

    # --- 3日連続下落後の反転 ---
    if len(close) >= 4:
        consecutive_down = all(close.iloc[-i - 1] < close.iloc[-i - 2] for i in range(3))
        if consecutive_down and change_pct > 0:
            net_score += _apply_signal("reversal_after_3down", fired_keys, fired_texts, weights)

    # --- 上昇確率計算 ---
    rise_probability = float(np.clip(50 + net_score * 0.6, 10, 92))

    # --- 反転検出 ---
    reversal_detected = (
        rsi < 30
        or (bb_position == "lower" and change_pct > 0)
        or (macd_signal == "bullish" and rsi < 45)
    )

    # --- 会社名取得 ---
    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        company_name = ticker

    # --- 総合判定文 ---
    if rise_probability >= 70:
        if reversal_detected:
            summary = "🚀 本日上昇の可能性が高く反転シグナルあり！強い買いタイミング"
        else:
            summary = "📈 本日上昇の可能性が高い。積極的に注目"
    elif rise_probability >= 55:
        summary = "🟢 やや上昇優位。指標を確認しながら検討"
    elif rise_probability >= 45:
        summary = "🟡 中立。様子見が無難"
    elif rise_probability >= 30:
        summary = "🟠 下落バイアスあり。押し目の深さに注意"
    else:
        summary = "🔴 弱気シグナル多数。下落リスクが高い"

    return StockAnalysis(
        ticker=ticker,
        company_name=company_name,
        current_price=current_price,
        change_pct=change_pct,
        rise_probability=rise_probability,
        signals=fired_texts,
        signal_keys=fired_keys,
        reversal_detected=reversal_detected,
        summary=summary,
        rsi=rsi,
        macd_signal=macd_signal,
        bb_position=bb_position,
        trend=trend,
        volume_surge=volume_surge,
        market=market,
        currency=currency,
    )


def format_analysis(analysis: StockAnalysis, show_weights: Optional[dict[str, float]] = None) -> str:
    """Discord用にフォーマットされた分析テキストを生成"""
    change_arrow = "▲" if analysis.change_pct >= 0 else "▼"
    trend_map = {"uptrend": "上昇トレンド📈", "downtrend": "下降トレンド📉", "sideways": "横ばい↔️"}

    if analysis.currency == "USD":
        price_str = f"${analysis.current_price:,.2f}"
        market_flag = "🇺🇸 米株"
    else:
        price_str = f"¥{analysis.current_price:,.0f}"
        market_flag = "🇯🇵 日本株"

    lines = [
        f"**{analysis.company_name}** ({analysis.ticker})  {market_flag}",
        f"現在値: {price_str}  {change_arrow}{abs(analysis.change_pct):.2f}%",
        f"トレンド: {trend_map.get(analysis.trend, '-')}",
        "",
        "**📊 テクニカル指標**",
        f"RSI(14): {analysis.rsi:.1f}",
        f"MACD: {'強気' if analysis.macd_signal == 'bullish' else '弱気' if analysis.macd_signal == 'bearish' else '中立'}",
        f"BB位置: {'下限(反発期待)' if analysis.bb_position == 'lower' else '上限(反落注意)' if analysis.bb_position == 'upper' else '中央'}",
        f"出来高急増: {'あり🔥' if analysis.volume_surge else 'なし'}",
        "",
        "**🎯 シグナル**",
    ]

    if analysis.signals:
        for sig_text, sig_key in zip(analysis.signals, analysis.signal_keys):
            w = show_weights.get(sig_key, 1.0) if show_weights else 1.0
            weight_tag = f" [重み:{w:.2f}]" if show_weights and abs(w - 1.0) > 0.05 else ""
            lines.append(f"  {sig_text}{weight_tag}")
    else:
        lines.append("  特筆シグナルなし")

    lines += [
        "",
        f"**⚡ 上昇確率: {analysis.rise_probability:.0f}%**",
        f"**{'🔄 反転シグナル検出！' if analysis.reversal_detected else ''}**".strip("*").strip() and
        f"**🔄 反転シグナル検出！**" if analysis.reversal_detected else "",
        f"**判定: {analysis.summary}**",
    ]
    # 空行を除去
    lines = [l for l in lines if l is not None]

    return "\n".join(lines)
