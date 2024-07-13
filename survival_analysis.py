import pandas as pd
from lifelines import CoxPHFitter
import pyodbc
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetch data from Azure SQL Database
def fetch_data():
    try:
        conn_str = f"Driver={{ODBC Driver 17 for SQL Server}};Server={os.getenv('SQL_SERVER')};Database={os.getenv('SQL_DATABASE')};UID={os.getenv('SQL_USERNAME')};PWD={os.getenv('SQL_PASSWORD')}"
        with pyodbc.connect(conn_str, timeout=10) as conn:
            query = "SELECT * FROM job_applications"
            df = pd.read_sql(query, conn)
            return df
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        raise

# Sort the records by application count in ascending order
def sort_records(df):
    df = df.sort_values(by='application_count', ascending=True)
    return df

# Send email using SendGrid
def send_email(subject, message):
    try:
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        email = Mail(
            from_email=os.getenv('SENDGRID_FROM_EMAIL'),  # Use environment variable for sender email
            to_emails=os.getenv('SENDGRID_TO_EMAIL'),  # Use environment variable for recipient email
            subject=subject,
            plain_text_content=message
        )
        response = sg.send(email)
        logging.info(f"Email sent. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# Run survival analysis
def run_survival_analysis():
    try:
        df = fetch_data()
        if df.empty:
            logging.warning("No data fetched for analysis.")
            return

        # Sort the records by application count in ascending order
        df = sort_records(df)

        # Get the latest application record for prediction
        latest_record = df.iloc[-1]

        # Previous records for training
        df = df.iloc[:-1]

        cph = CoxPHFitter()
        cph.fit(df, duration_col='time_to_response', event_col='event')
        summary = cph.summary.to_string()

        # Predict for the latest application
        latest_application_data = {
            'time_to_response': latest_record['time_to_response'],
            'match_experience': latest_record['match_experience'],
            'used_referral': latest_record['used_referral'],
            'contacted_employee': latest_record['contacted_employee'],
            'changed_resume': latest_record['changed_resume'],
        }
        latest_app_df = pd.DataFrame([latest_application_data])
        survival_function = cph.predict_survival_function(latest_app_df)
        survival_message = survival_function.to_string()

        # Send email notification
        subject = "Survival Analysis Results for the Latest Application"
        message = f"Survival Analysis Summary:\n{summary}\n\nSurvival Function for the Latest Application:\n{survival_message}"
        send_email(subject, message)
    except Exception as e:
        logging.error(f"Failed to run survival analysis: {e}")

if __name__ == "__main__":
    run_survival_analysis()