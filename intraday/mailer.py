import yagmail, os
from dotenv import load_dotenv
load_dotenv()

# Email sending logic
def send_email(df):
    body = df.to_markdown(index=False, floatfmt=".2f")
    yag = yagmail.SMTP(user=os.getenv("EMAIL_USER"),
                       password=os.getenv("EMAIL_PASS"),
                       host=os.getenv("SMTP_HOST"), port=int(os.getenv("SMTP_PORT")))
    yag.send(
        to=os.getenv("EMAIL_TO"),
        subject="\U0001F4C8 Intraday Picks – {}".format(df.at[0, 'SYMBOL'] if not df.empty else "No Picks"),
        contents=f"Hi,\n\nHere are today’s ideas:\n\n{body}\n\n—Your Trading Bot\n\nDisclaimer: This is not investment advice. For personal use only."
    )
