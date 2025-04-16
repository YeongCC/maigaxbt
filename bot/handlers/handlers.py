from __future__ import annotations

import asyncio
import base64
import html
import textwrap
import ast

from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
import pandas as pd

from api.config.application import PRODUCTION
from api.gpt.gpt_analysis import BinanceFuturesAnalyzer, FundFlowAnalyzer, multi_timeframe_analysis
from api.gpt.gpt_client import get_analysis, understand_user_prompt, async_generate_reply
from api.helpers.helper import async_get_crypto_price
from api.user.models import User, UserProfile
from api.wallet.mint_service import mint_xp_token
from bot.helper import async_request_chart, handle_unknown_coin
from bot.keyboards.keyboards import  up_down_kb
from bot.quries import add_bets_to_db, add_gen_data_to_db, add_report_data_to_db, add_xp_async, get_or_create_wallet, get_prompt, get_my_stats, get_wallet_if_exist, is_report_over_limit, is_user_over_limit, update_bet
import json
import logging


router = Router()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@router.message(Command(commands=["start"]))
async def handle_start_command(message: types.Message) -> None:
    if message.from_user is None:
        return
    user_id = message.from_user.id
    _, is_new = await User.objects.aget_or_create(
        pk=message.from_user.id,
        defaults={
            "username": message.from_user.username or f"user_{user_id}",
            "first_name": message.from_user.first_name or "",
            "last_name": message.from_user.last_name or "",
        },
    )
    
    if message.chat.type in ['group', 'supergroup']:
        new_text = textwrap.dedent("""\
            🎺 Welcome to MaigaXBT – the greatest trading AI, maybe ever. Some say the best!

            💡 What you can do:
            🔥 /analyse {token} – Powerful technical analysis, no fake news, just real insights.
            🔥 /report – Get full crypto report: price trend, RSI level, top OI movers, and fund flow summary!            
            🔥 /xpbalance – Check your XP. Because winners track their stats.
            🔥 NEW! Ask MaigaXBT anything about technical analysis on *any tradable token* — sharper than those so-called “experts.”
            🔥 Bet with MaigaXBT—Earn XP from your moves!

            Big trades, big wins—let’s make trading great again! 🚀💰
        """)

        new_text_welcome_back = textwrap.dedent("""\
            🎺 Welcome back to MaigaXBT – the greatest trading AI, maybe ever. Some say the best!
            
            💡 What you can do:
            🔥 /analyse {token} – Powerful technical analysis, no fake news, just real insights.
            🔥 /report – Get full crypto report: price trend, RSI level, top OI movers, and fund flow summary!            
            🔥 /xpbalance – Check your XP. Because winners track their stats.
            🔥 NEW! Ask MaigaXBT anything about technical analysis on *any tradable token* — sharper than those so-called “experts.”
            🔥 Bet with MaigaXBT—Earn XP from your moves!

            Big trades, big wins—let’s make trading great again! 🚀💰
        """)

        if is_new:
            await message.answer(new_text)
        else:
            await message.answer(new_text_welcome_back)
        return

    inline_wallet_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🪙 Create Wallet", callback_data="create_wallet")]
    ])

    new_text = textwrap.dedent("""\
        🎺 Welcome to MaigaXBT – the greatest trading AI, maybe ever. Some say the best!

        💡 What you can do:
        🔥 /analyse {token} – Powerful technical analysis, no fake news, just real insights.
        🔥 /report – Get full crypto report: price trend, RSI level, top OI movers, and fund flow summary!
        🔥 /xpbalance – Check your XP. Because winners track their stats.
        🔥 /createwallet – Create your Web3 wallet and instantly receive 1 XP token!
        🔥 NEW! Ask MaigaXBT anything about technical analysis on *any tradable token* — sharper than those so-called “experts.”
        🔥 Bet with MaigaXBT—Earn XP from your moves!

        Big trades, big wins—let’s make trading great again! 🚀💰
    """)

    new_text_welcome_back = textwrap.dedent("""\
        🎺 Welcome back to MaigaXBT – the greatest trading AI, maybe ever. Some say the best!

        💡 What you can do:
        🔥 /analyse {token} – Powerful technical analysis, no fake news, just real insights.
        🔥 /report – Get full crypto report: price trend, RSI level, top OI movers, and fund flow summary!
        🔥 /xpbalance – Check your XP. Because winners track their stats.
        🔥 /createwallet – Create your Web3 wallet and instantly receive 1 XP token!
        🔥 NEW! Ask MaigaXBT anything about technical analysis on *any tradable token* — sharper than those so-called “experts.”
        🔥 Bet with MaigaXBT—Earn XP from your moves!

        Big trades, big wins—let’s make trading great again! 🚀💰
    """)

    if is_new:
        await message.answer(new_text, reply_markup=inline_wallet_kb)
    else:
        await message.answer(new_text_welcome_back, reply_markup=inline_wallet_kb)

@router.callback_query(F.data == "create_wallet")
async def handle_wallet_button(callback: types.CallbackQuery):
    await callback.message.edit_reply_markup(reply_markup=None)
    if callback.message.chat.type in ['group', 'supergroup']:
        await callback.message.answer(
            "❌ Wallet can only be created in private chat.\n\n👉 [Click to PM me](https://t.me/maigaxbt_bot)",
            parse_mode="Markdown"
        )
        await callback.answer()
        return

    await process_create_wallet(callback.from_user.id, callback.message)
    await callback.answer()
    
@router.message(Command(commands=["createwallet"]))
async def handle_createwallet_command(message: types.Message) -> None:
    if message.from_user is None:
        return

    if message.chat.type in ['group', 'supergroup']:
        await message.reply(
            "❌ Creating a wallet can only be done in private chat.\n\n👉 [Click to PM me](https://t.me/maigaxbt_bot)",
            parse_mode="Markdown"
        )
        return

    await process_create_wallet(message.from_user.id, message)
    
async def process_create_wallet(user_id: int, message: types.Message):
    await message.answer("⏳ Checking your wallet status...")

    wallet, created = await get_or_create_wallet(user_id)

    if not created:
        await message.answer(
            f"💳 You already have a wallet:\n\n`{wallet.wallet_address}`",
            parse_mode="Markdown"
        )
        return

    await message.answer("⏳ Creating your wallet... Please wait...")

    profile = await UserProfile.objects.select_related('user').aget(user__id=user_id)

    try:

        tx_hash = await mint_xp_token(wallet.wallet_address, profile, 1)
        if tx_hash:
            await add_xp_async(profile, 1)
            if PRODUCTION:
                tx_url = f"https://opbnb.bscscan.com/tx/{tx_hash}"
            else:
                tx_url = f"https://opbnb-testnet.bscscan.com/tx/{tx_hash}"
            await message.answer(
            f"🎉 Your wallet has been successfully created!\n\n"
            f"💳 Wallet Address:\n`{wallet.wallet_address}`\n\n"
            f"🪙 You have received *1 XP token* as a welcome gift!\n"
            f"🔗 [View Transaction on BscScan]({tx_url})",
            parse_mode="Markdown"
            )

    except Exception as e:
        await message.answer(f"Wallet created but XP token minting failed.\n\nError: {e}")

@router.message(Command(commands=["xpbalance"]))
async def handle_xpbalance_command(message: types.Message) -> None:
    if message.from_user is None:
        return

    user_profile = await get_my_stats(user_id=message.from_user.id)
    await message.answer(f"Your balance is {user_profile.xp_points} XP")

@router.message(Command(commands=["analyse"]))
async def generate_response(message: types.Message) -> None:
    wallet = await get_wallet_if_exist(message.from_user.id)
    if not wallet:
        await message.reply("❌ You haven't created a wallet yet.\nPlease use /createwallet first.")
        return

    is_over, next_time = await is_user_over_limit(message.from_user.id)
    if is_over:
        next_time_str = next_time.strftime("%Y-%m-%d %H:%M:%S")
        await message.reply(
           f"⚠️ <b>Analysis Limit Reached</b> ⚠️\n"
           f"📊 You've reached the maximum of <b>20 analyses</b> per hour.\n"
           f"⏳ You can try again at: <b>{next_time_str}</b>\n"                      
        )
        return 
    
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.reply("Please provide a trading pair. Example: /analyse btc")
        return

    coin_symbol = args[1]
    
    coin_id = await handle_unknown_coin(message, coin_symbol)
    if coin_id is None:
        return

    await message.reply("⏳ Generating answer, please wait...")

    prompt = await get_prompt()
    
    chart_bytes, analysis_reply, token_price = await asyncio.gather(
        async_request_chart(coin_id, prompt.timeframe),
        get_analysis(symbol=coin_id, coin_name=coin_symbol.upper(), interval=prompt.timeframe, limit=120),
        async_get_crypto_price(coin_id)
    )

    bet_id =  await add_bets_to_db(user_id=message.from_user.id,
                             token=coin_id,
                             entry_price=token_price,
                             symbol=coin_symbol.upper())

    username = message.from_user.username or f"user_{message.from_user.id}"
    
    await message.reply_photo(
        photo=types.BufferedInputFile(chart_bytes, filename="chart.png"),
        reply_markup=up_down_kb(bet_id, message.from_user.id, username, message.chat.type),
        caption=html.escape(analysis_reply))

    await add_gen_data_to_db(analysis_reply, message.from_user.id)


@router.message(Command(commands=["report"]))
async def handle_report_command(message: types.Message):
    user_id = message.from_user.id
    is_limited, next_time = await is_report_over_limit(user_id)
    if is_limited:
        await message.answer(
            f"⚠️ You already generated a report in the past hour.\n"
            f"⏳ Try again after: {next_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📈 Market Behavior", callback_data="report_market")],
        [InlineKeyboardButton(text="📊 Open Interest Overview", callback_data="report_oi")],
        [InlineKeyboardButton(text="📉 Multi-Timeframe", callback_data="report_multi")],
        [InlineKeyboardButton(text="💸 Fund Flow Analysis", callback_data="report_fundflow")]
    ])
    await message.answer("Please select the analysis report to be generated:", reply_markup=keyboard)

@router.callback_query(F.data == "report_market")
async def handle_report_market(callback: CallbackQuery):
    await callback.message.answer("📈 You chose Market Behavior.")
    await callback.message.edit_reply_markup(reply_markup=None)
    
    analyzer = BinanceFuturesAnalyzer()
    symbols = analyzer.get_usdt_symbols()

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=s, callback_data=f"market_{s}")]
        # for s in symbols[:20]
        for s in symbols
    ])
    await callback.message.answer("Please select a symbol:", reply_markup=keyboard)
    await callback.answer()

@router.callback_query(F.data == "report_multi")
async def handle_report_multi(callback: CallbackQuery):
    await callback.message.answer("📉 You chose Multi-Timeframe Analysis.")    
    await callback.message.edit_reply_markup(reply_markup=None)    
    
    analyzer = BinanceFuturesAnalyzer()
    symbols = analyzer.get_usdt_symbols()
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=s, callback_data=f"multi_{s}")]
        # for s in symbols[:20]
        for s in symbols
    ])
    await callback.message.answer("Please select a symbol:", reply_markup=keyboard)
    await callback.answer()

@router.callback_query(F.data == "report_oi")
async def handle_report_oi(callback: CallbackQuery):
    await callback.message.answer("⏳ Generating Open Interest Report...")    
    await callback.message.edit_reply_markup(reply_markup=None)
    
    analyzer = BinanceFuturesAnalyzer()
    df = analyzer.analyze_positions()
    report_text = await analyzer.generate_position_report(df)
    # await add_report_data_to_db(callback.from_user.id, "oi", "ALL", report_text)
    await callback.message.answer(report_text, parse_mode="Markdown")

@router.callback_query(F.data == "report_fundflow")
async def handle_report_fundflow(callback: CallbackQuery):
    await callback.message.answer("⏳ Generating Fund Flow Report...")
    await callback.message.edit_reply_markup(reply_markup=None)    
    
    fund_analyzer = FundFlowAnalyzer()
    result = fund_analyzer.analyze_fund_flow()
    report_text = await result.get("analysis", "⚠️ Failed to generate fund flow analysis.")
    # await add_report_data_to_db(callback.from_user.id, "fundflow", "ALL", report_text)
    await callback.message.answer(report_text, parse_mode="Markdown")


@router.callback_query(F.data.startswith("market_"))
async def handle_market_symbol(callback: CallbackQuery):
    symbol = callback.data.split("market_")[1]
    
    await callback.message.answer(f"🔎 Analyzing Market Behavior for {symbol}...")
    await callback.message.edit_reply_markup(reply_markup=None)    
    
    analyzer = BinanceFuturesAnalyzer()
    position_data = analyzer.get_open_interest(symbol)
    if position_data and position_data["data"]:
        df = pd.DataFrame(position_data["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        current_oi = float(df["sumOpenInterest"].iloc[-1])
        percentile = (df["sumOpenInterest"].rank(pct=True).iloc[-1]) * 100
        changes = {}
        for period, hours in [("1hours", 1), ("4hours", 4), ("24hours", 24)]:
            past_oi = float(df[df["timestamp"] <= df["timestamp"].max() - pd.Timedelta(hours=hours)]
                            ["sumOpenInterest"].iloc[-1])
            change = current_oi - past_oi
            change_pct = (change / past_oi) * 100
            changes[period] = {"change": change, "change_percentage": change_pct}
        position_info = {
            "current_oi": current_oi,
            "percentile": percentile,
            "changes": changes
        }
        report_text = await analyzer.analyze_market_behavior(symbol, position_info)
    else:
        report_text = "⚠️ Failed to get market data."
    # await add_report_data_to_db(callback.from_user.id, "market", symbol, report_text)
    await callback.message.answer(report_text, parse_mode="Markdown")

@router.callback_query(F.data.startswith("multi_"))
async def handle_multi_symbol(callback: CallbackQuery):  
    symbol = callback.data.split("multi_")[1]
    await callback.message.answer(f"📉 Analyzing {symbol} on multiple timeframes...")
    await callback.message.edit_reply_markup(reply_markup=None)  
    
    result = await multi_timeframe_analysis(symbol)
    report_text = result.get("ai_analysis", "⚠️ Failed to generate multi-timeframe analysis.")
    # await add_report_data_to_db(callback.from_user.id, "multi", symbol, report_text)
    await callback.message.answer(report_text, parse_mode="Markdown")

@router.message(F.text)
async def handle_other_messages(message: types.Message) -> None:
    wallet = await get_wallet_if_exist(message.from_user.id)
    if not wallet:
        await message.reply("❌ You haven't created a wallet yet.\nPlease use /createwallet first.")
        return
    
    is_over, next_time = await is_user_over_limit(message.from_user.id)
    if is_over:
        next_time_str = next_time.strftime("%Y-%m-%d %H:%M:%S")
        await message.reply(
           f"⚠️ <b>Analysis Limit Reached</b> ⚠️\n"
           f"📊 You've reached the maximum of <b>20 analyses</b> per hour.\n"
           f"⏳ You can try again at: <b>{next_time_str}</b>\n"                
        )
        return 
    
    logging.debug(f"message: {repr(message.text)}")
    if message.chat.type in ['group', 'supergroup']:
        if not message.text.startswith('@maigaxbt'):
            return
        text_to_process = message.text[len('@maigaxbt'):].strip()
    else:
        text_to_process = message.text.strip()

    await message.reply("⏳ Generating answer, please wait...")
    
    prompt = await understand_user_prompt(text_to_process)
    logging.debug(f"prompt: ->{prompt}<-")
    try:
        dict_prompt = json.loads(prompt)
    except json.JSONDecodeError:
        await message.reply("⚠️ parsing failed，please check the input format")
        return
    
    symbol = dict_prompt.get("symbol", "").lstrip("$") 
    timeframe = dict_prompt["timeframe"]
    user_prompt = dict_prompt["user_prompt"]

    if not symbol:
        await message.reply(dict_prompt["none_existing_token"])
        return

    coin_id = await handle_unknown_coin(message, symbol)
    if coin_id is None:
        await message.reply(dict_prompt["none_existing_token"])
        return
    
    user_id = message.from_user.id
    username = message.from_user.username or f"user_{user_id}"
    chart_bytes, analysis_reply, token_price = await asyncio.gather(
        async_request_chart(coin_id, timeframe),
        get_analysis(symbol=coin_id, coin_name=symbol.upper(), interval=timeframe, limit=120, user_prompt=user_prompt),
        async_get_crypto_price(coin_id)
    )
    bet_id = await add_bets_to_db(
        user_id=user_id,
        token=coin_id,
        entry_price=token_price,
        symbol=symbol.upper()
    )
    
    if not chart_bytes:
        await message.reply(
            text=html.escape(analysis_reply),
            reply_markup=up_down_kb(bet_id, user_id, username, message.chat.type)
        )
    else:
        await message.reply_photo(
            photo=types.BufferedInputFile(chart_bytes, filename="chart.png"),
            reply_markup=up_down_kb(bet_id, user_id, username, message.chat.type),
            caption=html.escape(analysis_reply)
        )
        
    await add_gen_data_to_db(analysis_reply, message.from_user.id)


@router.message(F.photo)
async def handle_photo(message: types.Message):
    await message.reply("⏳ Generating answer, please wait...")
    file = await message.bot.get_file(message.photo[-1].file_id)
    file_content = await message.bot.download_file(file.file_path)

    img_base64 = base64.b64encode(file_content.read()).decode('utf-8')
    reply = await async_generate_reply(img_base64)

    await message.reply(reply)

@router.callback_query()
async def bet_up_or_down(callback: types.CallbackQuery) -> None:
    try:
        data_parts = callback.data.split(":")
        if len(data_parts) != 5:
            await callback.answer("Invalid callback data format.", show_alert=True)
            return

        up_down, bet_id, original_user_id, chat_type, username = data_parts
        original_user_id = int(original_user_id)

    except ValueError:
        await callback.answer("Error processing your request.", show_alert=True)
        return

    if chat_type in ["group", "supergroup"] and callback.from_user.id != original_user_id:
        await callback.answer("You are not allowed to vote on this prediction!", show_alert=True)
        return

    msg_id = callback.message.message_id
    if chat_type in ["group", "supergroup"]:
        chat_id = callback.message.chat.id
    else:
        chat_id = callback.from_user.id  

    if up_down in ["agree", "disagree"]:
        await update_bet(bet_id=bet_id, prediction=up_down, msg_id=msg_id, result="pending", chat_type=chat_type, chat_id=chat_id)

        await callback.message.edit_reply_markup(reply_markup=None)
        if chat_type in ["group", "supergroup"]:
            response_text = f"@{username} {up_down} this signal, @{username} please wait for your prediction result in the next hour!"
        else:
            response_text = f"You {up_down} this signal, please wait for your prediction result in the next hour!"


        await callback.message.reply(response_text, reply_markup=None)