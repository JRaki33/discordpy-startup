"""
Discord Bot - 低位株分析Bot（日本株・米株対応、予測追跡・精度改善機能付き）

コマンド一覧:
  /stock <コード>        - 個別銘柄を分析（例: /stock 7201.T, /stock AAPL）
  /screen                - 日本株の注目低位株TOP5を表示・予測保存
  /reversal              - 日本株の反転シグナル検出銘柄を表示・予測保存
  /usscreen              - 米株の注目銘柄TOP5を表示・予測保存
  /usreversal            - 米株の反転シグナル検出銘柄を表示・予測保存
  /report                - 日本株＋米株のデイリーレポートを手動実行
  /accuracy              - 予測精度レポートを表示
  /history <コード>      - 銘柄の予測履歴を表示
  /signals               - シグナル別的中率と重みを表示
  /watchlist             - 日本株監視リストを表示
  /uswatchlist           - 米株監視リストを表示
  /addstock <コード>     - 日本株監視リストに銘柄を追加
  /addusstock <コード>   - 米株監視リストに銘柄を追加
  /removestock <コード>  - 日本株監視リストから銘柄を削除
  /removeusstock <コード>- 米株監視リストから銘柄を削除
  /ping                  - 疎通確認
"""
import os
import traceback
import asyncio
from datetime import datetime, date, time as dtime
import discord
from discord.ext import commands, tasks

from stock_analyzer import analyze_stock, format_analysis
from stock_screener import (
    screen_low_price_stocks,
    screen_reversal_stocks,
    screen_us_low_price_stocks,
    screen_us_reversal_stocks,
    format_daily_report,
    DEFAULT_WATCHLIST,
    DEFAULT_US_WATCHLIST,
)
from prediction_db import (
    init_db,
    save_prediction,
    fill_results_for_date,
    update_signal_weights,
    load_signal_weights,
    get_accuracy_stats,
    get_ticker_history,
    get_pending_dates,
)

# --- Bot設定 ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)
token = os.environ["DISCORD_BOT_TOKEN"]
REPORT_CHANNEL_ID = int(os.environ.get("REPORT_CHANNEL_ID", "0"))

custom_watchlist: list[str] = list(DEFAULT_WATCHLIST)
custom_us_watchlist: list[str] = list(DEFAULT_US_WATCHLIST)


# --- 起動時 ---
@bot.event
async def on_ready():
    print(f"Bot起動: {bot.user}")
    init_db()
    print("DB初期化完了")
    if REPORT_CHANNEL_ID:
        daily_report_task.start()
        print(f"デイリーレポートタスク開始 → チャンネルID: {REPORT_CHANNEL_ID}")


# --- エラーハンドラ ---
@bot.event
async def on_command_error(ctx, error):
    orig_error = getattr(error, "original", error)
    error_msg = "".join(traceback.TracebackException.from_exception(orig_error).format())
    await ctx.send(f"エラーが発生しました:\n```{error_msg[:1800]}```")


# ===== 予測保存ヘルパー =====

def _record_analysis(analysis) -> None:
    """StockAnalysis を DB に保存する（同期）"""
    save_prediction(
        ticker=analysis.ticker,
        company_name=analysis.company_name,
        price=analysis.current_price,
        predicted_prob=analysis.rise_probability,
        reversal=analysis.reversal_detected,
        signal_keys=analysis.signal_keys,
        rsi=analysis.rsi,
        macd_signal=analysis.macd_signal,
        bb_position=analysis.bb_position,
        trend=analysis.trend,
        volume_surge=analysis.volume_surge,
    )


async def _fill_and_update(loop) -> tuple[int, int]:
    """未確定の予測に実績を埋め、重みを更新する（非同期ラッパー）"""
    pending = get_pending_dates()
    total_filled = 0
    for d in pending:
        n = await loop.run_in_executor(None, lambda dd=d: fill_results_for_date(dd))
        total_filled += n
    if total_filled > 0:
        await loop.run_in_executor(None, update_signal_weights)
    return len(pending), total_filled


# ===== デイリータスク =====

@tasks.loop(time=dtime(hour=8, minute=30))
async def daily_report_task():
    channel = bot.get_channel(REPORT_CHANNEL_ID)
    if channel is None:
        return

    loop = asyncio.get_event_loop()

    # 1) 前日以前の未確定予測に実績を埋めて重みを更新
    pending_count, filled_count = await _fill_and_update(loop)
    if filled_count > 0:
        await channel.send(
            f"📈 **昨日の予測結果を確認しました** ({filled_count}件)\n"
            "シグナル重みを自動更新しました。`/accuracy` で精度を確認できます。"
        )

    # 2) 最新の重みで分析・レポート送信
    await channel.send("📊 朝の分析を開始します。少々お待ちください...")
    await run_daily_report(channel, loop)


async def run_daily_report(channel, loop=None):
    """最新重みで日本株＋米株をスクリーニング → レポート送信 → 予測保存"""
    if loop is None:
        loop = asyncio.get_event_loop()

    weights = await loop.run_in_executor(None, load_signal_weights)

    jp_top = await loop.run_in_executor(
        None, lambda: screen_low_price_stocks(custom_watchlist, max_price=1000, top_n=5, weights=weights)
    )
    jp_reversal = await loop.run_in_executor(
        None, lambda: screen_reversal_stocks(custom_watchlist, max_price=1000, top_n=5, weights=weights)
    )
    us_top = await loop.run_in_executor(
        None, lambda: screen_us_low_price_stocks(custom_us_watchlist, max_price=50, top_n=5, weights=weights)
    )
    us_reversal = await loop.run_in_executor(
        None, lambda: screen_us_reversal_stocks(custom_us_watchlist, max_price=50, top_n=5, weights=weights)
    )

    report = format_daily_report(jp_top, jp_reversal, us_top, us_reversal)
    for chunk in split_message(report, 1900):
        await channel.send(chunk)

    seen = set()
    for a in jp_top + jp_reversal + us_top + us_reversal:
        if a.ticker not in seen:
            await loop.run_in_executor(None, lambda aa=a: _record_analysis(aa))
            seen.add(a.ticker)


# ===== コマンド =====

@bot.command(name="stock", help="銘柄を分析します (例: /stock 7201.T)")
async def stock_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/stock <ティッカー>` 例: `/stock 7201.T`")
        return

    ticker = ticker.upper()
    await ctx.send(f"🔍 **{ticker}** を分析中...")

    loop = asyncio.get_event_loop()
    weights = await loop.run_in_executor(None, load_signal_weights)
    analysis = await loop.run_in_executor(None, lambda: analyze_stock(ticker, weights))

    if analysis is None:
        await ctx.send(
            f"❌ `{ticker}` のデータを取得できませんでした。\n"
            "東証銘柄は末尾に `.T` を付けます（例: `7201.T`）"
        )
        return

    result = format_analysis(analysis, show_weights=weights)
    for chunk in split_message(result, 1900):
        await ctx.send(chunk)

    # 個別分析も予測として記録
    await loop.run_in_executor(None, lambda: _record_analysis(analysis))


@bot.command(name="screen", help="本日の注目低位株TOP5を表示")
async def screen_cmd(ctx):
    await ctx.send("📡 低位株をスクリーニング中...")
    loop = asyncio.get_event_loop()
    weights = await loop.run_in_executor(None, load_signal_weights)
    results = await loop.run_in_executor(
        None,
        lambda: screen_low_price_stocks(custom_watchlist, max_price=1000, top_n=5, weights=weights),
    )

    if not results:
        await ctx.send("現在、条件を満たす低位株は見つかりませんでした。")
        return

    await ctx.send(f"**🚀 本日の注目低位株 TOP{len(results)}**\n")
    for analysis in results:
        msg = format_analysis(analysis, show_weights=weights)
        for chunk in split_message(msg, 1900):
            await ctx.send(chunk)
        await loop.run_in_executor(None, lambda a=analysis: _record_analysis(a))
        await asyncio.sleep(0.5)


@bot.command(name="usscreen", help="米株の注目銘柄TOP5を表示 (例: /usscreen または /usscreen 30 で$30以下)")
async def usscreen_cmd(ctx, max_price: float = 50.0):
    await ctx.send(f"📡 米株をスクリーニング中（${max_price:.0f}以下）...")
    loop = asyncio.get_event_loop()
    weights = await loop.run_in_executor(None, load_signal_weights)
    results = await loop.run_in_executor(
        None,
        lambda: screen_us_low_price_stocks(custom_us_watchlist, max_price=max_price, top_n=5, weights=weights),
    )

    if not results:
        await ctx.send("現在、条件を満たす米株は見つかりませんでした。")
        return

    await ctx.send(f"**🇺🇸 本日の注目米株 TOP{len(results)}**\n")
    for analysis in results:
        msg = format_analysis(analysis, show_weights=weights)
        for chunk in split_message(msg, 1900):
            await ctx.send(chunk)
        await loop.run_in_executor(None, lambda a=analysis: _record_analysis(a))
        await asyncio.sleep(0.5)


@bot.command(name="usreversal", help="米株の反転シグナル検出銘柄を表示")
async def usreversal_cmd(ctx, max_price: float = 50.0):
    await ctx.send(f"🔄 米株の反転シグナル検出中（${max_price:.0f}以下）...")
    loop = asyncio.get_event_loop()
    weights = await loop.run_in_executor(None, load_signal_weights)
    results = await loop.run_in_executor(
        None,
        lambda: screen_us_reversal_stocks(custom_us_watchlist, max_price=max_price, top_n=5, weights=weights),
    )

    if not results:
        await ctx.send("現在、反転シグナルが出ている米株はありません。")
        return

    await ctx.send(f"**🇺🇸 米株 反転シグナル検出銘柄 ({len(results)}件)**\n")
    for analysis in results:
        msg = format_analysis(analysis, show_weights=weights)
        for chunk in split_message(msg, 1900):
            await ctx.send(chunk)
        await loop.run_in_executor(None, lambda a=analysis: _record_analysis(a))
        await asyncio.sleep(0.5)


@bot.command(name="reversal", help="反転シグナルが出ている低位株を表示")
async def reversal_cmd(ctx):
    await ctx.send("🔄 反転シグナル検出中...")
    loop = asyncio.get_event_loop()
    weights = await loop.run_in_executor(None, load_signal_weights)
    results = await loop.run_in_executor(
        None,
        lambda: screen_reversal_stocks(custom_watchlist, max_price=1000, top_n=5, weights=weights),
    )

    if not results:
        await ctx.send("現在、反転シグナルが出ている銘柄はありません。")
        return

    await ctx.send(f"**🔄 反転シグナル検出銘柄 ({len(results)}件)**\n")
    for analysis in results:
        msg = format_analysis(analysis, show_weights=weights)
        for chunk in split_message(msg, 1900):
            await ctx.send(chunk)
        await loop.run_in_executor(None, lambda a=analysis: _record_analysis(a))
        await asyncio.sleep(0.5)


@bot.command(name="report", help="デイリーレポートを手動実行")
async def report_cmd(ctx):
    loop = asyncio.get_event_loop()
    # 未確定予測の実績確認
    pending_count, filled_count = await _fill_and_update(loop)
    if filled_count > 0:
        await ctx.send(f"📈 {filled_count}件の予測結果を確認してシグナル重みを更新しました。")

    await ctx.send("📊 分析レポートを生成中...")
    await run_daily_report(ctx.channel, loop)


# ===== 精度・履歴コマンド =====

@bot.command(name="accuracy", help="予測精度レポートを表示")
async def accuracy_cmd(ctx):
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None, get_accuracy_stats)

    total = stats["total_predictions"]
    correct = stats["correct_predictions"]
    acc = stats["overall_accuracy"]

    lines = [
        "# 📊 予測精度レポート",
        "",
        f"**総予測数**: {total}件",
        f"**的中数**: {correct}件",
        f"**全体的中率**: **{acc:.1f}%**",
        "",
    ]

    if total < 10:
        lines.append("⚠️ データが少ないため精度が安定していません（10件以上で信頼性が上がります）")
    elif acc >= 60:
        lines.append("✅ 精度良好！シグナル重みが機能しています")
    elif acc >= 50:
        lines.append("🟡 まずまずの精度。データが蓄積されると改善されます")
    else:
        lines.append("🔴 精度が低め。シグナル重みを調整中です")

    lines += ["", "---", "**📈 直近の予測結果 (最新20件)**", ""]

    for r in stats["recent_results"][:10]:
        rose_icon = "✅" if r["rose"] else "❌"
        change = r["actual_change_pct"]
        change_str = f"{change:+.2f}%" if change is not None else "取得中"
        lines.append(
            f"{rose_icon} {r['predict_date']} **{r['ticker']}** "
            f"予測:{r['predicted_prob']:.0f}% 実績:{change_str}"
        )

    for chunk in split_message("\n".join(lines), 1900):
        await ctx.send(chunk)


@bot.command(name="signals", help="シグナル別的中率と現在の重みを表示")
async def signals_cmd(ctx):
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None, get_accuracy_stats)
    signal_stats = stats["signal_stats"]

    SIGNAL_LABELS = {
        "rsi_extreme_oversold":   "RSI極端売られすぎ(<25)",
        "rsi_oversold":           "RSI売られすぎ(<30)",
        "rsi_overbought":         "RSI高水準(>70)",
        "rsi_extreme_overbought": "RSI買われすぎ(>75)",
        "macd_golden_zero":       "MACDゴールデンクロス",
        "macd_histogram_rising":  "MACDヒスト上昇",
        "macd_dead_cross":        "MACDデッドクロス",
        "macd_falling":           "MACD下降",
        "bb_lower":               "BB下限タッチ",
        "bb_upper":               "BB上限タッチ",
        "ma_golden_cross":        "MA5/25ゴールデンクロス",
        "ma_dead_cross":          "MA5/25デッドクロス",
        "perfect_order_up":       "パーフェクトオーダー上昇",
        "perfect_order_down":     "パーフェクトオーダー下降",
        "volume_surge_up":        "出来高急増+上昇",
        "volume_surge_down":      "出来高急増+下落",
        "reversal_after_3down":   "3日連続下落後の反転",
    }

    lines = ["# 🎯 シグナル別的中率・重み", ""]

    if not signal_stats:
        lines.append("まだデータが不足しています（各シグナル5件以上必要）")
    else:
        lines.append("`シグナル名 | 的中率 | サンプル数 | 現在の重み`")
        lines.append("```")
        for s in signal_stats:
            acc = s["correct"] / s["total"] * 100 if s["total"] else 0
            label = SIGNAL_LABELS.get(s["signal_key"], s["signal_key"])
            w = s["weight"]
            trend = "↑" if w > 1.05 else ("↓" if w < 0.95 else "→")
            lines.append(
                f"{label:<28} {acc:5.1f}%  {s['total']:3d}件  {w:.2f}{trend}"
            )
        lines.append("```")
        lines.append("")
        lines.append("重み: >1.0=強化中 / <1.0=抑制中 / 1.0=初期値")

    for chunk in split_message("\n".join(lines), 1900):
        await ctx.send(chunk)


@bot.command(name="history", help="銘柄の予測履歴を表示 (例: /history 7201.T)")
async def history_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/history <ティッカー>` 例: `/history 7201.T`")
        return

    ticker = ticker.upper()
    loop = asyncio.get_event_loop()
    records = await loop.run_in_executor(None, lambda: get_ticker_history(ticker, limit=15))

    if not records:
        await ctx.send(f"`{ticker}` の予測記録がありません。")
        return

    lines = [f"# 📋 {ticker} の予測履歴", ""]
    correct = sum(
        1 for r in records
        if r["result_filled"]
        and ((r["predicted_prob"] >= 55 and r["rose"]) or (r["predicted_prob"] < 55 and not r["rose"]))
    )
    filled = sum(1 for r in records if r["result_filled"])
    if filled > 0:
        lines.append(f"直近{filled}件の的中率: **{correct/filled*100:.1f}%**")
        lines.append("")

    for r in records:
        date_str = r["predict_date"]
        prob = r["predicted_prob"]
        if r["result_filled"]:
            rose_icon = "✅" if r["rose"] else "❌"
            change = r["actual_change_pct"]
            result_str = f"{rose_icon} 実績: {change:+.2f}%"
        else:
            result_str = "⏳ 結果待ち"
        reversal_tag = " 🔄" if r["reversal"] else ""
        lines.append(f"`{date_str}` 予測:{prob:.0f}%{reversal_tag}  →  {result_str}")

    for chunk in split_message("\n".join(lines), 1900):
        await ctx.send(chunk)


@bot.command(name="checkresults", help="未確定の予測に実績を反映する（手動実行）")
async def checkresults_cmd(ctx):
    await ctx.send("🔍 過去の予測に実績を反映中...")
    loop = asyncio.get_event_loop()
    pending_count, filled_count = await _fill_and_update(loop)

    if filled_count == 0:
        await ctx.send(
            f"確認対象: {pending_count}日分\n"
            "新たに反映できた結果はありませんでした（株価データ未確定の可能性）"
        )
    else:
        await ctx.send(
            f"✅ {filled_count}件の実績を反映し、シグナル重みを更新しました。\n"
            "`/accuracy` で精度を確認、`/signals` で重みの変化を確認できます。"
        )


# ===== 監視リスト管理 =====

@bot.command(name="watchlist", help="日本株の監視銘柄リストを表示")
async def watchlist_cmd(ctx):
    lines = ["**📋 日本株 監視リスト**", ""]
    for ticker in custom_watchlist:
        lines.append(f"  • {ticker}")
    lines.append(f"\n合計: {len(custom_watchlist)}銘柄")
    await ctx.send("\n".join(lines))


@bot.command(name="uswatchlist", help="米株の監視銘柄リストを表示")
async def uswatchlist_cmd(ctx):
    lines = ["**📋 米株 監視リスト**", ""]
    for ticker in custom_us_watchlist:
        lines.append(f"  • {ticker}")
    lines.append(f"\n合計: {len(custom_us_watchlist)}銘柄")
    await ctx.send("\n".join(lines))


@bot.command(name="addstock", help="日本株監視リストに銘柄を追加 (例: /addstock 6758.T)")
async def addstock_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/addstock <ティッカー>` 例: `/addstock 6758.T`")
        return
    ticker = ticker.upper()
    if ticker in custom_watchlist:
        await ctx.send(f"`{ticker}` はすでに監視リストにあります。")
        return
    custom_watchlist.append(ticker)
    await ctx.send(f"✅ `{ticker}` を日本株監視リストに追加しました。（合計: {len(custom_watchlist)}銘柄）")


@bot.command(name="addusstock", help="米株監視リストに銘柄を追加 (例: /addusstock TSLA)")
async def addusstock_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/addusstock <ティッカー>` 例: `/addusstock TSLA`")
        return
    ticker = ticker.upper()
    if ticker in custom_us_watchlist:
        await ctx.send(f"`{ticker}` はすでに米株監視リストにあります。")
        return
    custom_us_watchlist.append(ticker)
    await ctx.send(f"✅ `{ticker}` を米株監視リストに追加しました。（合計: {len(custom_us_watchlist)}銘柄）")


@bot.command(name="removestock", help="日本株監視リストから銘柄を削除 (例: /removestock 6758.T)")
async def removestock_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/removestock <ティッカー>` 例: `/removestock 6758.T`")
        return
    ticker = ticker.upper()
    if ticker not in custom_watchlist:
        await ctx.send(f"`{ticker}` は日本株監視リストにありません。")
        return
    custom_watchlist.remove(ticker)
    await ctx.send(f"🗑️ `{ticker}` を削除しました。（残り: {len(custom_watchlist)}銘柄）")


@bot.command(name="removeusstock", help="米株監視リストから銘柄を削除 (例: /removeusstock TSLA)")
async def removeusstock_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/removeusstock <ティッカー>` 例: `/removeusstock TSLA`")
        return
    ticker = ticker.upper()
    if ticker not in custom_us_watchlist:
        await ctx.send(f"`{ticker}` は米株監視リストにありません。")
        return
    custom_us_watchlist.remove(ticker)
    await ctx.send(f"🗑️ `{ticker}` を削除しました。（残り: {len(custom_us_watchlist)}銘柄）")


@bot.command(name="ping", help="疎通確認")
async def ping_cmd(ctx):
    await ctx.send("pong 🏓")


# ===== ユーティリティ =====

def split_message(text: str, limit: int = 1900) -> list[str]:
    chunks = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


bot.run(token)
