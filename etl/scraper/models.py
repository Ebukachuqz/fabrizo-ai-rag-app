from dataclasses import dataclass

@dataclass
class Tweet:
    tweet_count: int
    tweet_id: str
    username: str
    text: str
    created_at: str
    url: str