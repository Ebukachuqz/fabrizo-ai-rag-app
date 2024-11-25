def clean_for_tts(response: str) -> str:
    """
    Clean response text by removing markdown formatting and citations
    for Text-to-Speech processing.
    
    Args:
        response (str): Original response with markdown formatting
        
    Returns:
        str: Cleaned text suitable for TTS
    """
    import re
    
    def remove_markdown_links(text):
        # Remove markdown links but keep the text
        # Matches pattern [text](url) and removes them
        return re.sub(r'\[.*?\]\(.*?\)', '', text)

    
    def remove_source_citations(text):
        # Remove entire source citation segments
        # Matches pattern "Source: [text](url)"
        return re.sub(r'\s*\|?\s*Source:\s*\[.*?\]\(.*?\)', '', text)
    
    def clean_special_characters(text):
        # Remove special characters and extra whitespace
        text = re.sub(r'[\[\]\|\{\}]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    cleaned_text = response
    cleaned_text = remove_markdown_links(cleaned_text)
    # cleaned_text = remove_source_citations(cleaned_text)
    # cleaned_text = clean_special_characters(cleaned_text)
    
    return cleaned_text