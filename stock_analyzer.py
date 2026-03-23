"""
低位株分析モジュール
テクニカル指標を使って上昇確率・反転シグナルを計算する
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional


@dataclass
class StockAnalysis:
    ticker: str
    company_name: str
    current_price: float
    change_pct: float
    rise_probability: float       # 0〜100%
    signals: list[str]            # シグナル一覧
    reversal_detected: bool       # 反転シグナルあり
    summary: str                  # 総合判定文
    rsi: float
    macd_signal: str              # "bullish" / "bearish" / "neutral"
    bb_position: str              # "lower" / "middle" / "upper"
    trend: str                    # "uptrend" / "downtrend" / "sideways"
    volume_surge: bool            # 出来高急増


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
    """RSI（相対力指数）を計算"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series):
    """MACD・シグナルライン・ヒストグラムを計算"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_bollinger(close: pd.Series, period: int = 20):
    """ボリンジャーバンドを計算"""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower


def analyze_stock(ticker: str) -> Optional[StockAnalysis]:
    """
    銘柄を分析して StockAnalysis を返す

    Args:
        ticker: yfinance 形式のティッカー (例: "9984.T")
    """
    df = fetch_stock_data(ticker)
    if df is None:
        return None

    close = df["Close"]
    volume = df["Volume"]

    # --- 各種指標計算 ---
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

    # 出来高急増（過去20日平均の2倍以上）
    vol_mean = volume.rolling(20).mean().iloc[-1]
    volume_surge = float(volume.iloc[-1]) >= vol_mean * 2.0

    # --- シグナル判定 ---
    signals = []
    bullish_score = 0  # 強気シグナル加点
    bearish_score = 0  # 弱気シグナル減点

    # RSI
    if rsi < 25:
        signals.append("🔴 RSI極端に売られすぎ(<25) → 反転期待")
        bullish_score += 30
    elif rsi < 30:
        signals.append("🟠 RSI売られすぎ(<30) → 買いシグナル")
        bullish_score += 20
    elif rsi > 75:
        signals.append("🔵 RSI買われすぎ(>75) → 反落注意")
        bearish_score += 20
    elif rsi > 70:
        signals.append("🔵 RSI高水準(>70) → 過熱注意")
        bearish_score += 10

    # MACD
    hist_now = float(macd_hist.iloc[-1])
    hist_prev = float(macd_hist.iloc[-2])
    macd_val = float(macd.iloc[-1])
    macd_sig_val = float(macd_sig.iloc[-1])

    if macd_val < 0 and hist_now > hist_prev and hist_now > 0:
        signals.append("✅ MACDゴールデンクロス(ゼロライン下) → 強い買いシグナル")
        bullish_score += 25
        macd_signal = "bullish"
    elif macd_val > 0 and hist_now > hist_prev:
        signals.append("✅ MACDヒストグラム上昇中 → 上昇継続")
        bullish_score += 15
        macd_signal = "bullish"
    elif macd_val > 0 and hist_now < 0 and hist_prev >= 0:
        signals.append("❌ MACDデッドクロス → 売りシグナル")
        bearish_score += 20
        macd_signal = "bearish"
    elif hist_now < hist_prev and hist_now < 0:
        signals.append("⚠️ MACD下降中")
        bearish_score += 10
        macd_signal = "bearish"
    else:
        macd_signal = "neutral"

    # ボリンジャーバンド
    bb_upper_val = float(bb_upper.iloc[-1])
    bb_lower_val = float(bb_lower.iloc[-1])
    bb_mid_val = float(bb_mid.iloc[-1])

    if current_price <= bb_lower_val:
        signals.append("✅ ボリンジャーバンド下限タッチ → 反発期待")
        bullish_score += 20
        bb_position = "lower"
    elif current_price >= bb_upper_val:
        signals.append("⚠️ ボリンジャーバンド上限 → 反落注意")
        bearish_score += 15
        bb_position = "upper"
    else:
        bb_position = "middle"

    # 移動平均線
    ma5_val = float(ma5.iloc[-1])
    ma25_val = float(ma25.iloc[-1])
    ma75_val = float(ma75.iloc[-1])
    ma5_prev = float(ma5.iloc[-2])
    ma25_prev = float(ma25.iloc[-2])

    if ma5_val > ma25_val and ma5_prev <= ma25_prev:
        signals.append("✅ 5日線が25日線をゴールデンクロス")
        bullish_score += 20
    elif ma5_val < ma25_val and ma5_prev >= ma25_prev:
        signals.append("❌ 5日線が25日線をデッドクロス")
        bearish_score += 20

    if current_price > ma25_val and ma25_val > ma75_val:
        signals.append("📈 移動平均パーフェクトオーダー(上昇)")
        bullish_score += 15
        trend = "uptrend"
    elif current_price < ma25_val and ma25_val < ma75_val:
        signals.append("📉 移動平均逆パーフェクトオーダー(下降)")
        bearish_score += 15
        trend = "downtrend"
    else:
        trend = "sideways"

    # 出来高急増
    if volume_surge:
        if change_pct > 0:
            signals.append("🔥 出来高急増+上昇 → 強い買い圧力")
            bullish_score += 15
        else:
            signals.append("⚠️ 出来高急増+下落 → 売り圧力注意")
            bearish_score += 10

    # 直近3日の連続下落後の反転
    if len(close) >= 4:
        consecutive_down = all(close.iloc[-i - 1] < close.iloc[-i - 2] for i in range(3))
        if consecutive_down and change_pct > 0:
            signals.append("🔄 3日連続下落後の反転 → 底打ちシグナル")
            bullish_score += 15

    # --- 上昇確率計算 ---
    # bullish_score(0〜100+) を 0〜100%にマッピング
    raw_score = bullish_score - bearish_score
    # -50〜+80くらいの範囲を 10〜90%にマッピング
    rise_probability = float(np.clip(50 + raw_score * 0.6, 10, 92))

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
        signals=signals,
        reversal_detected=reversal_detected,
        summary=summary,
        rsi=rsi,
        macd_signal=macd_signal,
        bb_position=bb_position,
        trend=trend,
        volume_surge=volume_surge,
    )


def format_analysis(analysis: StockAnalysis) -> str:
    """Discord用にフォーマットされた分析テキストを生成"""
    change_arrow = "▲" if analysis.change_pct >= 0 else "▼"
    trend_map = {"uptrend": "上昇トレンド📈", "downtrend": "下降トレンド📉", "sideways": "横ばい↔️"}

    lines = [
        f"**{analysis.company_name}** ({analysis.ticker})",
        f"現在値: ¥{analysis.current_price:,.0f}  {change_arrow}{abs(analysis.change_pct):.2f}%",
        f"トレンド: {trend_map.get(analysis.trend, '-')}",
        "",
        f"**📊 テクニカル指標**",
        f"RSI(14): {analysis.rsi:.1f}",
        f"MACD: {'強気' if analysis.macd_signal == 'bullish' else '弱気' if analysis.macd_signal == 'bearish' else '中立'}",
        f"BB位置: {'下限(反発期待)' if analysis.bb_position == 'lower' else '上限(反落注意)' if analysis.bb_position == 'upper' else '中央'}",
        f"出来高急増: {'あり🔥' if analysis.volume_surge else 'なし'}",
        "",
        f"**🎯 シグナル**",
    ]

    if analysis.signals:
        lines.extend(f"  {s}" for s in analysis.signals)
    else:
        lines.append("  特筆シグナルなし")

    lines += [
        "",
        f"**⚡ 上昇確率: {analysis.rise_probability:.0f}%**",
        f"**{'🔄 反転シグナル検出！' if analysis.reversal_detected else ''}**",
        f"**判定: {analysis.summary}**",
    ]

    return "\n".join(lines)
