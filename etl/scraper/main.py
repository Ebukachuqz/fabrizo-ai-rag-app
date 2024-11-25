import asyncio
from datetime import datetime, timedelta
import os
from csv_handler import initialize_csv, update_csv
from scraper import fetch_tweets

async def main():
    CSV_FILENAME = os.path.join(os.getcwd(), "data", "fabrizioromano_tweets.csv")

    current_date = datetime.now().strftime("%Y-%m-%d")
    QUERY = f'(from:FabrizioRomano) lang:en until:2024-10-30 since:2024-09-29 -filter:replies'

    initialize_csv(CSV_FILENAME)

    try:
        new_tweets = await fetch_tweets(QUERY, CSV_FILENAME)

        if new_tweets:
            update_csv(CSV_FILENAME, new_tweets)
            print(f"{datetime.now()} - Added {len(new_tweets)} new tweets to the CSV.")
        else:
            print(f"{datetime.now()} - No new tweets to add.")
    except Exception as e:
        print(f"{datetime.now()} - An error occurred: {e}")
    finally:
        cookies_path = "./cookies.json"
        if os.path.exists(cookies_path):
            try:
                os.remove(cookies_path)
                print(f"{datetime.now()} - Deleted {cookies_path}.")
            except Exception as delete_error:
                print(f"{datetime.now()} - Failed to delete {cookies_path}: {delete_error}")

if __name__ == "__main__":
    asyncio.run(main())
