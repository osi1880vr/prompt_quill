import nltk

# Sentence scoring (replace with your chosen criteria)
def sentence_score(sentence):
    # Example: Score based on word count and POS tags (optional)
    word_count = len(nltk.word_tokenize(sentence))
    important_words = sum(1 for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence))
                          if pos in ['NN', 'VB', 'JJ'])  # Nouns, verbs, adjectives (optional)
    return word_count + important_words  # Simple scoring example


def extractive_summary(text, num_sentences=3):
    """
    Creates a simple extractive summary of the text.

    Args:
        text: The input text to summarize.
        num_sentences: The desired number of sentences in the summary (default: 3).

    Returns:
        A string containing the extractive summary.
    """
    # Sentence tokenization
    try:
        sentences = nltk.sent_tokenize(text)

        scores = [sentence_score(sentence) for sentence in sentences]

        # Sort sentences by score (descending) and select top ones
        ranked_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)[:num_sentences]

        # Generate summary string
        summary = "\n".join([sentence for sentence, _ in ranked_sentences])
        return summary
    except:
        return text