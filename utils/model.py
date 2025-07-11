"""
Recipe Recommender Chatbot Model Module

This module provides functions to initialize the sentence transformer model and create embeddings for intent labels.
It is designed to be used in the recipe recommender chatbot application.

Contributor(s):
    Tyler Beetle <twbeetle@outlook.com>

Change Log
----------
2025-07-10 :: Refactored to improve logging, error handling, and modularity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

def compile_model_and_embeddings(intent_patterns):
    '''
    Initializes sentence transformer model and creates embeddings for intent labels

    Parameters
    ----------
    intent_patterns  : dict
        Dictionary of intent patterns

    Returns
    -------
    model  : SentenceTransformer
        Initialized sentence transformer model
    intent_embeddings  : dict
        Embeddings for intent labels
    intent_labels  : dict
        Labels for intent categories
    '''

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings for each intent category
    intent_embeddings = {}
    intent_labels = {}

    # Define the categories for which we will create embeddings
    for category in ['dietary', 'cuisine', 'meal_type', 'ingredients']:

        # Get the items for the current category from intent_patterns
        items = list(intent_patterns[category].keys())

        # If no items are found, use an empty list
        intent_labels[category] = items

        # Create embeddings for the items in the current category
        intent_embeddings[category] = (
            model.encode(items, show_progress_bar=False) if items else np.zeros((1, 384))
        )

    # Create embeddings for special intents (greeting, goodbye, help)
    intent_labels['special'] = ['greeting', 'goodbye', 'help']
    intent_embeddings['special'] = model.encode([
        "hello hi hey greetings",
        "bye goodbye exit see you",
        "help what can you do commands options"
    ], show_progress_bar=False)

    return model, intent_embeddings, intent_labels