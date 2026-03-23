"""
Discord Bot - 低位株分析Bot
コマンド一覧:
  /stock <コード>    - 個別銘柄を分析（例: /stock 7201.T）
  /screen            - 低位株の本日注目TOP5を表示
  /reversal          - 反転シグナル検出銘柄を表示
  /watchlist         - 現在の監視リストを表示
  /addstock <コード> - 監視リストに銘柄を追加
  /report            - デイリーレポートを手動実行
  /ping              - 疎通確認
"""
import os
import traceback
import asyncio
from datetime import datetime, time as dtime
import discord
from discord.ext import commands, tasks

from stock_analyzer import analyze_stock, format_analysis
from stock_screener import (
    screen_low_price_stocks,
    screen_reversal_stocks,
    format_daily_report,
    DEFAULT_WATCHLIST,
)

# --- Bot設定 ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)
token = os.environ["DISCORD_BOT_TOKEN"]

# 毎朝レポートを送るチャンネルID（環境変数から取得）
# 設定されていない場合は自動レポートをスキップ
REPORT_CHANNEL_ID = int(os.environ.get("REPORT_CHANNEL_ID", "0"))

# ユーザーが追加した銘柄（メモリ上で保持。再起動でリセット）
custom_watchlist: list[str] = list(DEFAULT_WATCHLIST)


# --- エラーハンドラ ---
@bot.event
async def on_command_error(ctx, error):
    orig_error = getattr(error, "original", error)
    error_msg = "".join(traceback.TracebackException.from_exception(orig_error).format())
    await ctx.send(f"エラーが発生しました:\n```{error_msg[:1800]}```")


# --- 起動時 ---
@bot.event
async def on_ready():
    print(f"Bot起動: {bot.user}")
    if REPORT_CHANNEL_ID:
        daily_report_task.start()
        print(f"デイリーレポートタスク開始 → チャンネルID: {REPORT_CHANNEL_ID}")


# --- デイリーレポートタスク（毎朝8:30） ---
@tasks.loop(time=dtime(hour=8, minute=30))
async def daily_report_task():
    channel = bot.get_channel(REPORT_CHANNEL_ID)
    if channel is None:
        return
    await channel.send("📊 朝の分析を開始します。少々お待ちください...")
    await run_daily_report(channel)


async def run_daily_report(channel):
    """非同期で分析を実行してレポートを送信"""
    loop = asyncio.get_event_loop()
    # 分析は同期処理なのでスレッドで実行
    top = await loop.run_in_executor(
        None, lambda: screen_low_price_stocks(custom_watchlist, max_price=1000, top_n=5)
    )
    reversal = await loop.run_in_executor(
        None, lambda: screen_reversal_stocks(custom_watchlist, max_price=1000, top_n=5)
    )
    report = format_daily_report(top, reversal)
    # 2000文字制限対応で分割送信
    for chunk in split_message(report, 1900):
        await channel.send(chunk)


# --- コマンド: 個別銘柄分析 ---
@bot.command(name="stock", help="銘柄を分析します (例: /stock 7201.T)")
async def stock_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/stock <ティッカー>` 例: `/stock 7201.T`")
        return

    ticker = ticker.upper()
    await ctx.send(f"🔍 **{ticker}** を分析中... しばらくお待ちください")

    loop = asyncio.get_event_loop()
    analysis = await loop.run_in_executor(None, lambda: analyze_stock(ticker))

    if analysis is None:
        await ctx.send(
            f"❌ `{ticker}` のデータを取得できませんでした。\n"
            "ティッカーを確認してください（東証銘柄は末尾に `.T` を付けます。例: `7201.T`）"
        )
        return

    result = format_analysis(analysis)
    for chunk in split_message(result, 1900):
        await ctx.send(chunk)


# --- コマンド: 低位株スクリーニング ---
@bot.command(name="screen", help="本日の注目低位株TOP5を表示")
async def screen_cmd(ctx):
    await ctx.send("📡 低位株をスクリーニング中... しばらくお待ちください")
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: screen_low_price_stocks(custom_watchlist, max_price=1000, top_n=5),
    )

    if not results:
        await ctx.send("現在、条件を満たす低位株は見つかりませんでした。")
        return

    await ctx.send(f"**🚀 本日の注目低位株 TOP{len(results)}**\n")
    for analysis in results:
        msg = format_analysis(analysis)
        for chunk in split_message(msg, 1900):
            await ctx.send(chunk)
        await asyncio.sleep(0.5)


# --- コマンド: 反転シグナル ---
@bot.command(name="reversal", help="反転シグナルが出ている低位株を表示")
async def reversal_cmd(ctx):
    await ctx.send("🔄 反転シグナル検出中...")
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: screen_reversal_stocks(custom_watchlist, max_price=1000, top_n=5),
    )

    if not results:
        await ctx.send("現在、反転シグナルが出ている銘柄はありません。")
        return

    await ctx.send(f"**🔄 反転シグナル検出銘柄 ({len(results)}件)**\n")
    for analysis in results:
        msg = format_analysis(analysis)
        for chunk in split_message(msg, 1900):
            await ctx.send(chunk)
        await asyncio.sleep(0.5)


# --- コマンド: 手動デイリーレポート ---
@bot.command(name="report", help="デイリーレポートを手動実行")
async def report_cmd(ctx):
    await ctx.send("📊 分析レポートを生成中... しばらくお待ちください")
    await run_daily_report(ctx.channel)


# --- コマンド: 監視リスト表示 ---
@bot.command(name="watchlist", help="現在の監視銘柄リストを表示")
async def watchlist_cmd(ctx):
    lines = ["**📋 現在の監視リスト**", ""]
    for ticker in custom_watchlist:
        lines.append(f"  • {ticker}")
    lines.append(f"\n合計: {len(custom_watchlist)}銘柄")
    await ctx.send("\n".join(lines))


# --- コマンド: 銘柄追加 ---
@bot.command(name="addstock", help="監視リストに銘柄を追加 (例: /addstock 6758.T)")
async def addstock_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/addstock <ティッカー>` 例: `/addstock 6758.T`")
        return

    ticker = ticker.upper()
    if ticker in custom_watchlist:
        await ctx.send(f"`{ticker}` はすでに監視リストにあります。")
        return

    custom_watchlist.append(ticker)
    await ctx.send(f"✅ `{ticker}` を監視リストに追加しました。（合計: {len(custom_watchlist)}銘柄）")


# --- コマンド: 銘柄削除 ---
@bot.command(name="removestock", help="監視リストから銘柄を削除 (例: /removestock 6758.T)")
async def removestock_cmd(ctx, ticker: str = None):
    if ticker is None:
        await ctx.send("使い方: `/removestock <ティッカー>` 例: `/removestock 6758.T`")
        return

    ticker = ticker.upper()
    if ticker not in custom_watchlist:
        await ctx.send(f"`{ticker}` は監視リストにありません。")
        return

    custom_watchlist.remove(ticker)
    await ctx.send(f"🗑️ `{ticker}` を監視リストから削除しました。（残り: {len(custom_watchlist)}銘柄）")


# --- コマンド: ping ---
@bot.command(name="ping", help="疎通確認")
async def ping_cmd(ctx):
    await ctx.send("pong 🏓")


# --- ユーティリティ ---
def split_message(text: str, limit: int = 1900) -> list[str]:
    """長いテキストを Discord の文字数制限で分割"""
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
