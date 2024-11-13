import re

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["                   # Start of character range
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002B50-\U00002B55"  # Additional stars, etc.
        "\U00002300-\U000023FF"  # Additional symbols
        "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

