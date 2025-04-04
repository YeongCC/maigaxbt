from __future__ import annotations

import asyncio
from datetime import timedelta
from aiogram import Bot
from asgiref.sync import async_to_sync
from celery import Celery, shared_task
from celery.schedules import crontab

from api.config.application import OPBNB_PROVIDER_RPC_URL
from api.config.celery import tg_bot_token
from django.utils.timezone import now
from django.core.exceptions import ObjectDoesNotExist
from django.utils.timezone import localtime
from api.config import celery as config
from api.helpers.helper import get_crypto_prices
from api.user.models import Bet, Transaction, Wallet
from api.wallet.mint_service import mint_xp_token
from web3 import Web3
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Celery("main")
app.config_from_object(config)
app.autodiscover_tasks()

w3 = Web3(Web3.HTTPProvider(OPBNB_PROVIDER_RPC_URL))

app.conf.beat_schedule = {
    'verify-bets-every-1-hour': {
        'task': 'tasks.app.verify_bets',
        'schedule': crontab(minute='*/57'),  # Run every 1 hour
    },
    'check-transactions-every-1-minute': {
        'task': 'tasks.app.check_failed_transactions', # Run every 1 minute
        'schedule': crontab(minute='*'),
    },
    'check-success-transactions-every-second': {
        'task': 'tasks.app.check_success_transactions', # Run every 1 second
        'schedule': timedelta(seconds=1)
    },
}


@shared_task
def verify_bets():
    bets = Bet.objects.filter(result='pending')
    coins_ids = bets.values_list('token', flat=True)
    coins_dict = get_crypto_prices(coins_ids)

    for bet in bets:
        verification_time = bet.created_at + timedelta(hours=bet.verification_time)    
        if now() >= verification_time:
            result = bet.check_bet_result(coins_dict[bet.token])

            if result:
                xp_reward, user_profile, msg_id, receiver_id, chat_type, result_message = result
                
                try:
                    wallet = Wallet.objects.get(user=user_profile)
                except ObjectDoesNotExist:
                    send_telegram_message.delay(
                        receiver_id,
                        "Your wallet was not found! Please create your wallet first.",
                        msg_id,
                    )             
                    logging.error(f"[Minted Error] Wallet not found for {user_profile.user.username}")
                    continue  

                tx_hash = async_to_sync(mint_xp_token)(wallet.wallet_address, user_profile, xp_reward)

                if tx_hash:
                    tx_url = f"https://testnet.bscscan.com/tx/{tx_hash}"

                    full_message = (
                        f"{result_message}\n\n"
                        f"🪙 Symbol: {bet.symbol}\n"
                        f"🕒 Bet Time: {localtime(bet.created_at).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"🔗 [View Transaction]({tx_url})"
                    )
                        
                    send_telegram_message.delay(receiver_id, full_message, msg_id)
                    
                    logging.info(f"Minted {xp_reward} XP to {user_profile.user.username} | TX: {tx_url}")
                else:
                    logging.error(f"[Minted Error] Failed to mint XP for {user_profile.user.username}")


    return f"Verified {bets.count()} bets"



@shared_task
def check_failed_transactions():
    failed_txs = Transaction.objects.filter(status='failed', tx_hash__isnull=True)
    updated = 0
    retried = 0

    for tx in failed_txs:
        try:
            user = tx.user
            wallet_address = tx.wallet.wallet_address
            amount = tx.amount

            for attempt in range(3):
                try:
                    tx_hash = async_to_sync(mint_xp_token)(wallet_address, user, float(amount))
                    if tx_hash:
                        tx.tx_hash = tx_hash
                        tx.status = "pending"
                        tx.retry_count += 1
                        tx.updated_at = now()
                        tx.save()
                        updated += 1
                        logging.info(f"[RETRY-MINT SUCCESS] tx updated: {tx_hash}")
                        break
                except Exception as e:
                    logging.warning(f"[RETRY-MINT FAILED ATTEMPT {attempt+1}] {e}")
                    tx.retry_count += 1
                    asyncio.sleep(1)
                    continue

            tx.save()
            retried += 1

        except Exception as e:
            logging.error(f"[TASK ERROR] Failed to retry mint for tx id={tx.id}: {e}")

    logging.info(f"Retried {retried} failed txs, updated {updated} with new tx_hash.")
    return f"{updated} transaction(s) updated from failed retry"

@shared_task
def check_success_transactions():

    pending_txs = Transaction.objects.filter(status='pending').exclude(tx_hash__isnull=True)
    updated = 0

    for tx in pending_txs:
        try:
            receipt = w3.eth.get_transaction_receipt(tx.tx_hash)
            if receipt is not None:
                if receipt.status == 1:
                    tx.status = 'success'
                    explorer_url = f"https://opbnb-testnet.bscscan.com/tx/{tx.tx_hash}"
                    msg = (
                        f"✅ Your XP mint transaction has been confirmed on-chain!\n\n"
                        f"🧾 Hash: `{tx.tx_hash}`\n"
                        f"🔗 [View on BscScan]({explorer_url})"
                    )
                else:
                    tx.status = 'failed'
                tx.updated_at = now()
                tx.save()
                updated += 1
        except Exception as e:
            print(f"Error checking tx {tx.tx_hash}: {e}")
            continue

    print(f"Checked {pending_txs.count()} pending txs, updated {updated}")
    return f"{updated} transaction(s) updated"


async def _send(user_id, message, msg_id):
    bot = Bot(token=tg_bot_token)

    await asyncio.sleep(0.5)  # Delay to prevent rate limiting
    await bot.send_message(user_id, message, request_timeout=10, reply_to_message_id=msg_id)
    await bot.session.close()  # Ensure the session is closed


@shared_task(bind=True)
def send_telegram_message(self, user_id, message, msg_id):
    try:
        async_to_sync(_send)(user_id, message, msg_id)  # Wrap the async call in async_to_sync

    except Exception as e:
        self.retry(exc=e, countdown=5, max_retries=3)
