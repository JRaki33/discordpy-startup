"""
予測記録・精度管理モジュール

SQLite に予測を保存し、翌営業日の実績と照合して
シグナルごとの的中率を追跡・重みに反映する。

テーブル構成:
  predictions   - 毎日の予測レコード
  signal_stats  - シグナル別の的中率と動的重み
"""
import sqlite3
import json
import os
from datetime import date, datetime, timedelta
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get("STOCK_DB_PATH", "stock_predictions.db")

# デフォルトの重み（1.0 = 変化なし, >1.0 = 強め, <1.0 = 弱め）
DEFAULT_WEIGHTS: dict[str, float] = {
    "rsi_extreme_oversold": 1.0,
    "rsi_oversold": 1.0,
    "rsi_overbought": 1.0,
    "macd_golden_zero": 1.0,
    "macd_histogram_rising": 1.0,
    "macd_dead_cross": 1.0,
    "macd_falling": 1.0,
    "bb_lower": 1.0,
    "bb_upper": 1.0,
    "ma_golden_cross": 1.0,
    "ma_dead_cross": 1.0,
    "perfect_order_up": 1.0,
    "perfect_order_down": 1.0,
    "volume_surge_up": 1.0,
    "volume_surge_down": 1.0,
    "reversal_after_3down": 1.0,
}

# 重みの上下限（外れ値防止）
WEIGHT_MIN = 0.3
WEIGHT_MAX = 2.5
# 重み更新に必要な最低サンプル数
MIN_SAMPLES_FOR_UPDATE = 10


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """DB とテーブルを初期化（冪等）"""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_date  TEXT NOT NULL,          -- 予測日 (YYYY-MM-DD)
                ticker        TEXT NOT NULL,
                company_name  TEXT,
                price         REAL NOT NULL,           -- 予測時の終値
                predicted_prob REAL NOT NULL,          -- 上昇確率 (0-100)
                reversal      INTEGER NOT NULL,        -- 反転シグナルあり (0/1)
                signal_keys   TEXT NOT NULL,           -- JSON 配列
                rsi           REAL,
                macd_signal   TEXT,
                bb_position   TEXT,
                trend         TEXT,
                volume_surge  INTEGER,
                -- 翌日実績（後から埋める）
                result_date   TEXT,
                result_price  REAL,
                actual_change_pct REAL,
                rose          INTEGER,                 -- 1=上昇, 0=下落/横ばい
                result_filled INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_pred_date
                ON predictions (predict_date);
            CREATE INDEX IF NOT EXISTS idx_pred_ticker
                ON predictions (ticker);

            CREATE TABLE IF NOT EXISTS signal_stats (
                signal_key    TEXT PRIMARY KEY,
                total         INTEGER DEFAULT 0,
                correct       INTEGER DEFAULT 0,
                weight        REAL DEFAULT 1.0,
                updated_at    TEXT
            );
        """)
        # デフォルト重みを signal_stats に INSERT（未登録のみ）
        for key, w in DEFAULT_WEIGHTS.items():
            conn.execute(
                "INSERT OR IGNORE INTO signal_stats (signal_key, weight) VALUES (?, ?)",
                (key, w),
            )


def save_prediction(
    ticker: str,
    company_name: str,
    price: float,
    predicted_prob: float,
    reversal: bool,
    signal_keys: list[str],
    rsi: float,
    macd_signal: str,
    bb_position: str,
    trend: str,
    volume_surge: bool,
    predict_date: Optional[date] = None,
) -> int:
    """予測を DB に保存して row id を返す"""
    if predict_date is None:
        predict_date = date.today()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
              (predict_date, ticker, company_name, price, predicted_prob,
               reversal, signal_keys, rsi, macd_signal, bb_position, trend, volume_surge)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                predict_date.isoformat(),
                ticker,
                company_name,
                price,
                predicted_prob,
                int(reversal),
                json.dumps(signal_keys),
                rsi,
                macd_signal,
                bb_position,
                trend,
                int(volume_surge),
            ),
        )
        return cur.lastrowid


def fill_results_for_date(target_date: date) -> int:
    """
    target_date の予測に対して実績（翌営業日終値）を埋める。
    yfinance から取得して result_filled=1 にする。
    戻り値: 更新した件数
    """
    import yfinance as yf

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, ticker, price FROM predictions "
            "WHERE predict_date = ? AND result_filled = 0",
            (target_date.isoformat(),),
        ).fetchall()

    if not rows:
        return 0

    updated = 0
    # 翌日以降のデータを取得（最大5営業日分取得して最初のデータを使う）
    lookup_end = target_date + timedelta(days=7)
    today = date.today()
    if lookup_end > today:
        lookup_end = today

    if lookup_end <= target_date:
        return 0

    for row in rows:
        pred_id, ticker, pred_price = row["id"], row["ticker"], row["price"]
        try:
            df = yf.Ticker(ticker).history(
                start=(target_date + timedelta(days=1)).isoformat(),
                end=lookup_end.isoformat(),
            )
            if df.empty:
                continue
            result_price = float(df["Close"].iloc[0])
            result_date_val = df.index[0].date().isoformat()
            actual_change = (result_price - pred_price) / pred_price * 100
            rose = 1 if result_price > pred_price else 0

            with get_db() as conn:
                conn.execute(
                    """
                    UPDATE predictions
                    SET result_date=?, result_price=?, actual_change_pct=?,
                        rose=?, result_filled=1
                    WHERE id=?
                    """,
                    (result_date_val, result_price, actual_change, rose, pred_id),
                )
            updated += 1
        except Exception:
            continue

    return updated


def update_signal_weights():
    """
    実績が埋まった予測をもとにシグナル別の的中率を再計算し、
    重みを更新する。
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT signal_keys, predicted_prob, rose "
            "FROM predictions WHERE result_filled = 1"
        ).fetchall()

    # signal_key ごとに (total, correct) を集計
    stats: dict[str, list[int]] = {k: [0, 0] for k in DEFAULT_WEIGHTS}

    for row in rows:
        keys = json.loads(row["signal_keys"])
        # "上昇予測" とは predicted_prob >= 55 とする
        predicted_up = row["predicted_prob"] >= 55
        actually_rose = bool(row["rose"])
        correct = (predicted_up == actually_rose)

        for k in keys:
            if k not in stats:
                stats[k] = [0, 0]
            stats[k][0] += 1
            if correct:
                stats[k][1] += 1

    now_str = datetime.now().isoformat(timespec="seconds")
    with get_db() as conn:
        for key, (total, correct) in stats.items():
            if total < MIN_SAMPLES_FOR_UPDATE:
                conn.execute(
                    "INSERT OR REPLACE INTO signal_stats (signal_key, total, correct, weight, updated_at) "
                    "VALUES (?, ?, ?, "
                    "  COALESCE((SELECT weight FROM signal_stats WHERE signal_key=?), 1.0), "
                    "  ?)",
                    (key, total, correct, key, now_str),
                )
                continue

            accuracy = correct / total
            # 精度0.5基準で重みを線形調整 (0.5 → 1.0, 0.7 → 1.4, 0.3 → 0.6)
            new_weight = max(WEIGHT_MIN, min(WEIGHT_MAX, accuracy * 2.0))
            conn.execute(
                "INSERT OR REPLACE INTO signal_stats "
                "(signal_key, total, correct, weight, updated_at) VALUES (?, ?, ?, ?, ?)",
                (key, total, correct, new_weight, now_str),
            )


def load_signal_weights() -> dict[str, float]:
    """DB からシグナル重みを読み込む"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT signal_key, weight FROM signal_stats"
        ).fetchall()
    weights = dict(DEFAULT_WEIGHTS)
    for row in rows:
        weights[row["signal_key"]] = row["weight"]
    return weights


def get_accuracy_stats() -> dict:
    """全体精度とシグナル別精度をまとめて返す"""
    with get_db() as conn:
        total_preds = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE result_filled=1"
        ).fetchone()[0]

        correct_preds = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE result_filled=1 "
            "AND ((predicted_prob >= 55 AND rose=1) OR (predicted_prob < 55 AND rose=0))"
        ).fetchone()[0]

        signal_rows = conn.execute(
            "SELECT signal_key, total, correct, weight FROM signal_stats "
            "WHERE total >= 5 ORDER BY CAST(correct AS REAL)/total DESC"
        ).fetchall()

        recent = conn.execute(
            "SELECT predict_date, ticker, company_name, predicted_prob, "
            "actual_change_pct, rose FROM predictions "
            "WHERE result_filled=1 ORDER BY result_date DESC LIMIT 20"
        ).fetchall()

    overall_accuracy = (correct_preds / total_preds * 100) if total_preds else 0
    return {
        "total_predictions": total_preds,
        "correct_predictions": correct_preds,
        "overall_accuracy": overall_accuracy,
        "signal_stats": [dict(r) for r in signal_rows],
        "recent_results": [dict(r) for r in recent],
    }


def get_ticker_history(ticker: str, limit: int = 10) -> list[dict]:
    """銘柄ごとの予測履歴を返す"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT predict_date, predicted_prob, reversal, actual_change_pct, rose, result_filled "
            "FROM predictions WHERE ticker=? ORDER BY predict_date DESC LIMIT ?",
            (ticker.upper(), limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_dates() -> list[date]:
    """実績未確定の予測日一覧を返す（古い順）"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT predict_date FROM predictions "
            "WHERE result_filled=0 ORDER BY predict_date"
        ).fetchall()
    today = date.today()
    return [
        date.fromisoformat(r["predict_date"])
        for r in rows
        if date.fromisoformat(r["predict_date"]) < today
    ]
