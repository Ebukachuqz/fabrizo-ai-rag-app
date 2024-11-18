def get_rating_from_emoji(emoji):
    emoji_ratings = {
        '😞': 1,  # Very unsatisfied
        '🙁': 2,  # Unsatisfied
        '😐': 3,  # Neutral
        '🙂': 4,  # Satisfied
        '😀': 5   # Very satisfied
    }
    return emoji_ratings.get(emoji, 3)  # Default to 3 if emoji not found