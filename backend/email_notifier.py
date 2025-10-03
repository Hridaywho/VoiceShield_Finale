import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EmailNotifier:
	def __init__(self):
		# Using Gmail's SMTP server
		self.smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
		self.smtp_port = int(os.getenv("EMAIL_SMTP_PORT", 587))

		# Credentials: prefer environment variables for safety
		self.sender_email = os.getenv("EMAIL_SENDER", "hridayd1908@gmail.com")
		self.app_password = os.getenv("EMAIL_APP_PASSWORD","dgtuqxxgaajinfvp")

		if not self.app_password:
			logger.warning("EMAIL_APP_PASSWORD is not set. Sending may fail until it's provided via environment variable.")

	def send_alert(self, recipient_email, anger_level, location="Not Specified"):
		try:
			msg = MIMEMultipart()
			msg["From"] = self.sender_email
			msg["To"] = recipient_email
			msg["Subject"] = "⚠️ ALERT: High Aggression Detected!"

			# Email body with HTML formatting
			current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			body = f"""
			<html>
			<body style="font-family: Arial, sans-serif; line-height: 1.6;">
				<div style="max-width: 600px; margin: 0 auto; padding: 20px;">
					<h2 style="color: #dc2626;">🚨 Aggression Alert</h2>
					<p>High levels of aggression have been detected by Voice Shield.</p>
                    
					<div style="background: #fef2f2; border-left: 4px solid #dc2626; padding: 15px; margin: 20px 0;">
						<p style="margin: 0;"><strong>Details:</strong></p>
						<ul style="margin: 10px 0;">
							<li>Anger Level: <strong>{anger_level}%</strong></li>
							<li>Time: {current_time}</li>
							<li>Location: {location}</li>
						</ul>
					</div>
                    
					<p style="color: #4b5563; font-size: 0.9em;">
						This is an automated alert from your Voice Shield system. 
						Please take appropriate action if necessary.
					</p>
				</div>
			</body>
			</html>
			"""

			msg.attach(MIMEText(body, "html"))

			# Connect to SMTP server and send email
			with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
				server.starttls()
				server.login(self.sender_email, self.app_password)
				server.send_message(msg)

			logger.info(f"Alert email sent successfully to {recipient_email}")
			return True

		except Exception as e:
			logger.error(f"Failed to send alert email: {str(e)}")
			return False


# Test the email functionality if run directly
if __name__ == "__main__":
	# Set up logging
	logging.basicConfig(level=logging.INFO)

	# Create notifier instance
	notifier = EmailNotifier()

	# Test sending an email
	print("Testing email notification...")
	success = notifier.send_alert(
		recipient_email="ayush.fireball2005@gmail.com",
		anger_level=75.5,
		location="Test Location",
	)

	if success:
		print("✅ Test email sent successfully!")
	else:
		print("❌ Failed to send test email. Check your credentials and internet connection.")