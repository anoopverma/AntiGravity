import os
import requests
import logging

logger = logging.getLogger(__name__)

class TelegramAlerter:
    """
    A simple Telegram Alerter using the official Telegram Bot API.
    To get your API key:
    1. Open Telegram and search for "BotFather".
    2. Start a chat and send the command "/newbot" to create a new bot.
    3. Follow instructions to get your HTTP API bot token.
    4. To get your Chat ID, search for your new bot in Telegram, start a chat, and send it a message.
    5. Then visit: https://api.telegram.org/bot<YourBOTToken>/getUpdates
    6. Find the "chat":{"id": <YourChatID>} in the response.
    
    Then, add these to your .env file:
    TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
    TELEGRAM_CHAT_ID=YOUR_CHAT_ID
    """
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            logger.warning("Telegram alerter disabled. To enable, add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to your .env file.")

    def send_alert(self, message_text):
        if not self.enabled:
            return False
            
        try:
            full_msg = f"🚨 *AntiGravity Bot ALERT* 🚨\n\n{message_text}"
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                "chat_id": self.chat_id,
                "text": full_msg,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Telegram alert sent successfully.")
                return True
            else:
                logger.error(f"❌ Failed to send Telegram alert. API Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Exception sending Telegram alert: {e}")
            return False
