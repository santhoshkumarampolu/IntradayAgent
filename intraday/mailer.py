import yagmail, os
from dotenv import load_dotenv
load_dotenv()

# Email sending logic
def send_email(df):
    # Ensure at least 1 to 5 picks are sent (if available)
    picks = df.head(5) if not df.empty else df
    # Prepare a minimal, actionable table
    display_df = picks.rename(columns={
        'TckrSymb': 'Stock',
        'ClsPric': 'Current Price',
        'entry': 'Entry',
        'target': 'Target',
    })
    # Only keep relevant columns
    columns_to_show = [col for col in ['Stock', 'Current Price', 'Entry', 'Target'] if col in display_df.columns]
    display_df = display_df[columns_to_show]
    # Add trading time info
    display_df['Start Time'] = '09:15 AM'
    display_df['Exit Time'] = '03:15 PM'
    # Reorder columns
    display_df = display_df[['Stock', 'Current Price', 'Entry', 'Target', 'Start Time', 'Exit Time']]
    # Add Bhavcopy date to the email body
    bhavcopy_date = None
    if not picks.empty and 'TradDt' in picks.columns:
        bhavcopy_date = picks['TradDt'].iloc[0]
    elif not picks.empty and 'BizDt' in picks.columns:
        bhavcopy_date = picks['BizDt'].iloc[0]
    date_line = f"Based on Bhavcopy date: {bhavcopy_date}" if bhavcopy_date else "Date not available"
    body = f"{date_line}\n\n" + display_df.to_markdown(index=False, floatfmt=".2f")
    symbol_col = 'Stock' if 'Stock' in display_df.columns else (display_df.columns[0] if not display_df.empty else None)
    subject_symbol = display_df.at[0, symbol_col] if (not display_df.empty and symbol_col) else "No Picks"
    yag = yagmail.SMTP(user=os.getenv("EMAIL_USER"),
                       password=os.getenv("EMAIL_PASS"),
                       host=os.getenv("SMTP_HOST"), port=int(os.getenv("SMTP_PORT")))
    yag.send(
        to=os.getenv("EMAIL_TO"),
        subject="\U0001F4C8 Intraday Picks – {}".format(subject_symbol),
        contents=f"Hi,\n\nHere are today’s ideas (Top 1-5):\n\n{body}\n\n—Your Trading Bot\n\nDisclaimer: This is not investment advice. For personal use only."
    )
