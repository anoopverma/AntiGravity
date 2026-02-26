import os
import urllib.parse
import requests
import logging

logger = logging.getLogger(__name__)

class WhatsAppAlerter:
    """
    A simple WhatsApp Alerter using the free CallMeBot API.
    To get your API key:
    1. Add +34 693 28 14 06 to your Phone Contacts (Name it "CallMeBot").
    2. Send this message to the contact: "I allow callmebot to send me messages"
    3. The bot will automatically reply with your unique apikey.
    
    Then, add these to your .env file:
    WHATSAPP_PHONE=+919876543210  (Your number with country code)
    WHATSAPP_API_KEY=YOUR_API_KEY
    """
    def __init__(self):
        self.phone = os.getenv("WHATSAPP_PHONE", "").strip()
        self.api_key = os.getenv("WHATSAPP_API_KEY", "").strip()
        self.enabled = bool(self.phone and self.api_key)
        
        if not self.enabled:
            logger.warning("WhatsApp alerter disabled. To enable, add WHATSAPP_PHONE and WHATSAPP_API_KEY to your .env file.")

    def send_alert(self, message_text):
        if not self.enabled:
            return False
            
        try:
            # Prefix the message so we know it's coming from the bot
            full_msg = f"🚨 *AntiGravity Bot ALERT* 🚨\n\n{message_text}"
            encoded_message = urllib.parse.quote(full_msg)
            
            # Note: Do not include the '+' in the phone number for the URL payload
            phone_clean = self.phone.replace("+", "")
            
            url = f"https://api.callmebot.com/whatsapp.php?phone={phone_clean}&text={encoded_message}&apikey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.info("✅ WhatsApp specific error alert sent successfully.")
                return True
            else:
                logger.error(f"❌ Failed to send WhatsApp alert. API Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Exception sending WhatsApp alert: {e}")
            return False
