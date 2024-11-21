import asyncio
from datetime import datetime
from random import randint
from typing import List
from twikit import TooManyRequests
from httpx import ConnectTimeout, ReadTimeout
from models import Tweet
from twitter_client import get_twitter_client

async def fetch_tweets(query: str, csv_filename: str, max_retries: int = 3, retry_delay: int = 5) -> List[Tweet]:
    """Fetch tweets matching the query"""
    from csv_handler import get_first_tweet_id  # Import here to avoid circular import
    
    client = get_twitter_client()
    tweets = None
    new_tweets: List[Tweet] = []
    first_tweet_id = get_first_tweet_id(csv_filename)
    print(f"First tweet ID from CSV: {first_tweet_id}")

    while True:
        try:
            if tweets is None:
                print(f"{datetime.now()} - Fetching initial tweets...")
                tweets = await client.search_tweet(query=query, product='Latest')
            else:
                wait_time = randint(14, 23)
                print(f"{datetime.now()} - Fetching next batch after {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                tweets = await tweets.next()

            for tweet in tweets:
                if str(tweet.id) == first_tweet_id:
                    print(f"{datetime.now()} - Stopping as we found the first tweet ID.")
                    return new_tweets

                new_tweets.append(Tweet(
                    tweet_count=len(new_tweets) + 1,
                    tweet_id=str(tweet.id),
                    username=tweet.user.name,
                    text=tweet.text,
                    created_at=str(tweet.created_at),
                    url=f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
                ))
                print(f"{datetime.now()} - Collected tweet ID: {tweet.id}")

        except TooManyRequests as e:
            rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
            wait_time = (rate_limit_reset - datetime.now()).total_seconds()
            print(f"{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}.")
            await asyncio.sleep(wait_time)
        except (ConnectTimeout, ReadTimeout):
            max_retries -= 1
            if max_retries <= 0:
                print(f"{datetime.now()} - Max retries reached. Exiting.")
                break
            print(f"{datetime.now()} - Timeout. Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            print(f"{datetime.now()} - Unexpected error: {e}")
            break

    return new_tweets