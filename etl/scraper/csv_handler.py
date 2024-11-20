import os
import csv
import pandas as pd
from typing import List, Optional
from models import Tweet

def initialize_csv(csv_filename: str) -> None:
    """Create CSV file with headers if it doesn't exist"""
    if not os.path.exists(csv_filename):
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['tweet_count', 'tweet_id', 'username', 'text', 'created_at', 'url'])

def get_first_tweet_id(csv_filename: str) -> Optional[str]:
    """Get the ID of the first tweet in the CSV"""
    try:
        df = pd.read_csv(csv_filename)
        return str(df['tweet_id'].iloc[0]) if not df.empty else None
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

def update_csv(csv_filename: str, new_tweets: List[Tweet]) -> None:
    """Update CSV with new tweets"""
    if not new_tweets:
        return

    try:
        with open(csv_filename, 'r') as file:
            existing_data = file.readlines()
    except FileNotFoundError:
        existing_data = []

    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        if existing_data:
            writer.writerow(existing_data[0].strip().split(','))
        else:
            writer.writerow(['tweet_count', 'tweet_id', 'username', 'text', 'created_at', 'url'])

        for tweet in new_tweets:
            writer.writerow([
                tweet.tweet_count,
                tweet.tweet_id,
                tweet.username,
                tweet.text,
                tweet.created_at,
                tweet.url
            ])

        if existing_data:
            file.writelines(existing_data[1:])