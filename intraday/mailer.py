import yagmail, os
from dotenv import load_dotenv
load_dotenv()

# Email sending logic
def send_email(df):
    # Ensure at least 1 to 5 picks are sent (if available)
    picks = df.head(5) if not df.empty else df
    if picks.empty or len(picks.columns) == 0:
        body = "No valid picks for today."
        subject_symbol = "No Picks"
    else:
        # Prepare a minimal, actionable table
        display_df = picks.rename(columns={
            'TckrSymb': 'Stock',
            'ClsPric': 'Current Price',
            'entry': 'Entry',
            'target': 'Target',
        })
        # Only keep relevant columns if they exist
        columns_to_show = [col for col in ['Stock', 'Current Price', 'Entry', 'Target'] if col in display_df.columns]
        display_df['Start Time'] = '09:15 AM'
        display_df['Exit Time'] = '03:15 PM'
        columns_to_show += ['Start Time', 'Exit Time']
        display_df = display_df[columns_to_show]
        # Add Bhavcopy date to the email body
        bhavcopy_date = None
        if not picks.empty and 'TradDt' in picks.columns:
            bhavcopy_date = picks['TradDt'].iloc[0]
        elif not picks.empty and 'BizDt' in picks.columns:
            bhavcopy_date = picks['BizDt'].iloc[0]
        date_line = f"Based on Bhavcopy date: {bhavcopy_date}" if bhavcopy_date else "Date not available"
        # --- Detailed rationale for each pick ---
        rationales = []
        for _, row in picks.iterrows():
            rationale = f"{row['TckrSymb']}: "
            rationale += f"RSI={row.get('RSI', 'N/A'):.1f}, " if 'RSI' in row else ''
            rationale += f"EMA20={'above' if row.get('EMA20',0) > row.get('EMA50',0) else 'below'} EMA50, " if 'EMA20' in row and 'EMA50' in row else ''
            rationale += f"ADX={row.get('ADX', 'N/A'):.1f}, " if 'ADX' in row else ''
            rationale += f"Sector strength={'strong' if row.get('confidence',0)>0.7 else 'neutral/weak'}, " if 'confidence' in row else ''
            rationale += f"ML confidence={row.get('confidence', 'N/A'):.2f}, " if 'confidence' in row else ''
            rationale += f"Recent event detected, " if row.get('has_event', False) else ''
            rationale += f"No negative news."
            rationales.append(rationale.strip().rstrip(','))
        rationale_text = '\n'.join(rationales)
        body = f"{date_line}\n\n" + display_df.to_markdown(index=False, floatfmt=".2f") + "\n\nTrade Rationales (in simple terms):\n" + rationale_text
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
