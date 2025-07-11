"""
Recipe Recommender Chatbot Intent Recognition Module

This module provides functions to recognize user intents based on input text.
It uses semantic similarity with embeddings and regex patterns to identify dietary preferences,
cuisines, meal types, time constraints, ingredients, and special intents.

Contributor(s):
    Tyler Beetle <twbeetle@outlook.com>

Change Log
----------
2025-07-10 :: Refactored to improve logging, error handling, and modularity.
"""
import re
from numpy.linalg import norm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Define custom filler phrases
FILLER_PHRASES = [
    "give me", "show me", "can i have", "i want", "i'd like", "please", "could you", "would you", "may i have", "let me have", "find me", "get me", "i need", "i would like", "i'm looking for", "do you have", "i'd love", "i'd want", "i wish for", "i wish to have", "i wish to get", "i wish to see", "recipe", "recipes"
]

# Function to clean user input
def clean_user_input(user_input):
    cleaned = user_input.lower()

    # Remove filler phrases
    for phrase in FILLER_PHRASES:
        cleaned = re.sub(r'\b' + re.escape(phrase) + r'\b', '', cleaned)

    # Remove stopwords
    words = [w for w in re.findall(r'\w+', cleaned) if w not in ENGLISH_STOP_WORDS]

    return ' '.join(words)

def recognize_intent(
    user_input,
    intent_patterns,
    intent_embeddings,
    intent_labels,
    intent_model
):
    '''
    Recognizes intents from user input using semantic similarity and regex patterns

    Parameters
    ----------
    user_input  : str
        User's input text
    intent_patterns  : dict
        Regex patterns for intents
    intent_embeddings  : dict
        Embeddings for intent labels
    intent_labels  : dict
        Labels for intent categories
    intent_model  : SentenceTransformer
        Sentence transformer model for encoding

    Returns
    -------
    intents  : dict
        Recognized intents with matched categories
    '''

    def best_match(user_emb ,ref_embs , labels, threshold, topn):
        """Finds the best matching labels based on cosine similarity."""

        # Ensure ref_embs is a 2D array
        if len(ref_embs) == 0:
            return []
    
        # Calculate cosine similarity between user embedding and reference embeddings
        sims = (ref_embs @ user_emb) / (norm(ref_embs, axis=1) * norm(user_emb) + 1e-8)

        # Get topn indices based on similarity scores
        top_idx = sims.argsort()[::-1][:topn]

        # Filter out labels with similarity below the threshold
        return [(labels[i], sims[i]) for i in top_idx if sims[i] > threshold]
    
    # Clean user input to remove stopwords and filler phrases
    user_input_cleaned = clean_user_input(user_input)
    # Validate user_input
    if not isinstance(user_input_cleaned, str) or not user_input_cleaned.strip():
        raise ValueError("user_input must be a non-empty string after cleaning")
    
    # Encode user input to get its embedding
    user_emb = intent_model.encode([user_input_cleaned], show_progress_bar=False)[0]
    intents = {
        'dietary': [], 'cuisine': [], 'meal_type': [],
        'time_constraint': [], 'ingredients': [], 'special': None
    }

    # Initialize matched flag
    matched = False

    # Check for special intents (greeting, goodbye, help)
    special_match = best_match(
        user_emb, intent_embeddings['special'], intent_labels['special'], threshold=0.5, topn=1
    )

    # If a special intent is matched, assign it
    if special_match:
        intents['special'] = special_match[0][0]
        matched = True

    # Check for regex patterns in user input (time_constraint)
    for key, pattern in intent_patterns['time_constraint'].items():
        # Check if the pattern matches the user input
        if re.search(pattern, user_input, re.IGNORECASE):
            # If it matches, add to intents and set matched flag
            intents['time_constraint'].append(key)
            matched = True

    words = set(user_input_cleaned.lower().strip().split())
    all_matches = {}

    # Check for dietary, cuisine, meal type, and ingredients using best_match
    for category in ['dietary', 'cuisine', 'meal_type', 'ingredients']:
        matches = best_match(
            user_emb, intent_embeddings[category], intent_labels[category], threshold=0.7, topn=10
        )
        all_matches[category] = [label for label, _ in matches]

    assigned_labels = set()
    user_input_lower = user_input_cleaned.lower()

    # Check if any of the words in user input match the labels
    for category in ['cuisine', 'ingredients', 'meal_type', 'dietary']:

        # Check if any of the labels match the words in user input
        for label in all_matches[category]:

            # Check if the label matches any word in user input or is a substring
            if (
                label.lower() in words
                or any(word in label.lower() for word in words)
                or label.lower() in user_input_lower
            ):
                # If it matches, add to intents and set matched flag
                if label not in assigned_labels:
                    intents[category].append(label)
                    assigned_labels.add(label)
                    matched = True

    # If intents for dietary, cuisine, meal_type, or ingredients are empty,
    # assign all matches from the respective category
    # This ensures that if no specific intent was matched, it still provides relevant labels
        if not intents[category] and all_matches[category]:
            for label in all_matches[category]:
                if label not in assigned_labels:
                    intents[category].append(label)
                    assigned_labels.add(label)
                    matched = True

    intents['matched'] = matched

    return intents