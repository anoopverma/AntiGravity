import os
import urllib.parse
import requests
import logging

logger = logging.getLogger(__name__)

class WhatsAppAlerter:
    """
    A standard WhatsApp Alerter using the official Meta WhatsApp Cloud API.
    To use this, you need to set up a Meta Developer App:
    1. Go to developers.facebook.com and create an app (type: Business).
    2. Add the WhatsApp product to your app.
    3. Get your temporary or permanent Access Token.
    4. Get your specific Phone Number ID (from the Meta dashboard).
    
    Then, add these to your .env file:
    META_WA_TOKEN=YOUR_META_ACCESS_TOKEN
    META_PHONE_ID=YOUR_PHONE_NUMBER_ID
    WHATSAPP_PHONE=919876543210  (The destination number you want to alert, with country code, NO '+')
    """
    def __init__(self):
        self.phone = os.getenv("WHATSAPP_PHONE", "").strip().replace("+", "")
        self.api_token = os.getenv("META_WA_TOKEN", "").strip()
        self.phone_id = os.getenv("META_PHONE_ID", "").strip()
        
        self.enabled = bool(self.phone and self.api_token and self.phone_id)
        
        if not self.enabled:
            logger.warning("WhatsApp alerter disabled. To enable, add META_WA_TOKEN, META_PHONE_ID, and WHATSAPP_PHONE to your .env.")

    def send_alert(self, message_text):
        if not self.enabled:
            return False
            
        try:
            full_msg = f"🚨 *AntiGravity Bot ALERT* 🚨\n\n{message_text}"
            
            url = f"https://graph.facebook.com/v17.0/{self.phone_id}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": self.phone,
                "type": "text",
                "text": {
                    "body": full_msg
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("✅ WhatsApp specific error alert sent successfully.")
                return True
            else:
                logger.error(f"❌ Failed to send WhatsApp alert. API Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Exception sending WhatsApp alert: {e}")
            return False
