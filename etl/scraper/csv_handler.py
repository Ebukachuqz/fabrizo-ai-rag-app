import os
import csv
import pandas as pd
from typing import List, Optional
from models import Tweet

def initialize_csv(csv_filename: str) -> None:
    """Create CSV file with headers if it doesn't exist"""
    print("In initialize_csv")
    if not os.path.exists(csv_filename):
        print(f'{csv_filename} does not exist. Creating it now...')
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
    """Update CSV with new tweets at the start, recalculating serial numbers."""
    if not new_tweets:
        return

    try:
        with open(csv_filename, 'r') as file:
            existing_data = list(csv.reader(file))
    except FileNotFoundError:
        existing_data = []

    header = existing_data[0] if existing_data else ['tweet_count', 'tweet_id', 'username', 'text', 'created_at', 'url']
    existing_data = existing_data[1:] if existing_data else []

    all_data = [
        [
            index + 1,  # reassign serial number
            tweet.tweet_id,
            tweet.username,
            tweet.text,
            tweet.created_at,
            tweet.url
        ]
        for index, tweet in enumerate(
            new_tweets + [Tweet(*row) for row in existing_data] 
        )
    ]

    # Write everything back to the file
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(all_data)