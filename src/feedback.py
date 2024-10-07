import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

mongo_url = os.getenv("MONGO_DB_URL")
client = pymongo.MongoClient(mongo_url)
db = client['fabrizoai_db']
feedback_collection = db['feedback']

def store_feedback(query, answer, rating):
    feedback_data = {
        'query': query,
        'answer': answer,
        'rating': rating
    }
    feedback_collection.insert_one(feedback_data)
    print("Feedback stored successfully")
