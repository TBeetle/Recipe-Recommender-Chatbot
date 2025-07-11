"""
Recipe Recommender Chatbot

Main module for running the recipe recommender chatbot, which processes user input
and generates recipe recommendations based on intents.

Contributor(s):
    Tyler Beetle <twbeetle@outlook.com>

Change Log
----------
2025-07-10 :: Refactored to improve logging, error handling, and modularity.
"""

import logging
from logging.handlers import RotatingFileHandler
import pandas as pd

from utils.preprocess import preprocess_data, build_intent_patterns
from utils.model import compile_model_and_embeddings
from utils.intent import recognize_intent
from utils.filter import filter_recipes, format_recipe

# Path to the cleaned recipes dataset
data_path = 'data/CLEANED_recipes.csv'

# Configure logging with rotation to prevent file bloat
handler = RotatingFileHandler(
    'conversation.log', maxBytes=1000000, backupCount=5
)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_recommendations(user_input):
    '''
    Generates recipe recommendations based on user input

    Parameters
    ----------
    user_input  : str
        User's input text

    Returns
    -------
    output  : str
        Formatted response with recipe recommendations or appropriate message
    '''
    logging.info(f"Processing user input (length: {len(user_input)} characters)")
    
    # Get the intents by passing user input to the intent recognition function
    intents = recognize_intent(
        user_input, intent_patterns, intent_embeddings, intent_labels, intent_model
    )

    # Handle special intents (greeting, goodbye)
    if intents['special']:
        output = (
            "Hello! 👋 How can I help you find recipes today?"

            if intents['special'] == 'greeting'
            else "Goodbye! 👋 Happy cooking!"
        )
        logging.info(f"Returning special response: {intents['special']}")

        return output

    # Filter recipes based on intents
    filtered_recipes = filter_recipes(intents, user_input, recipes_df)

    # Handle unmatched intents with conversational responses
    if not intents.get('matched'):
        user_input_lower = user_input.lower()
        responses = {
            'your name': "I'm Rice, your recipe assistant! 😊",
            'how are you': "I'm just code, but I'm ready to cook up some recipes!",
            'thank': "You're welcome! Want another recipe?",
            'shoes': "No shoes here, but I've got tasty recipes!"
        }

        for key, response in responses.items():
            if key in user_input_lower:
                logging.info(f"Returning conversational response for '{key}'")

                return response
        output = (
            "I'm here to help you find recipes! Try asking for a cuisine, "
            "ingredient, or meal type, like 'Italian chicken' or 'quick dessert'."
        )
        logging.info("Returning fallback conversational response")

        return output

    # Handle case where no recipes match
    if filtered_recipes.empty:
        logging.info("No recipes matched the criteria")

        return "Sorry, I couldn't find any recipes matching your request."

    # Format and return top 3 recipes
    output = "\n".join(
        format_recipe(row, i + 1) for i, row in filtered_recipes.head(3).iterrows()
    )
    logging.info(f"Returning {min(3, len(filtered_recipes))} recipe recommendations")

    return output

def run_chatbot():
    '''
    Runs the recipe recommender chatbot in an interactive loop

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    print("🍳 RECIPE RECOMMENDER CHATBOT 🍳\nType 'quit' to exit.")
    
    while True:
        try:
            # Get user input and log it
            user_input = input("You: ").strip()
            logging.info(f"Received user input: {user_input[:50]}...")
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit']:
                print("Recipe Recommender: Goodbye! 👋")
                logging.info("Chatbot session ended by user")
                break
            
            # Get recommendations based on user input
            response = get_recommendations(user_input)
            print(f"Recipe Recommender: {response}")

        except KeyboardInterrupt:
            print("\nRecipe Recommender: Goodbye! 👋")
            logging.info("Chatbot session interrupted by user")
            break

        except Exception as e:
            print(f"Error: {str(e)}. Please try again.")
            logging.error(f"Unexpected error during chat: {str(e)}")

if __name__ == "__main__":

    try:
        # Load data and model once at startup for performance
        recipes_df = pd.read_csv(data_path)
        recipes_df = preprocess_data(recipes_df)
        intent_patterns = build_intent_patterns(recipes_df)
        intent_model, intent_embeddings, intent_labels = compile_model_and_embeddings(intent_patterns)
        logging.info("Chatbot initialized successfully")

        # Start the chatbot
        run_chatbot()

    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        logging.error(f"Initialization error: {str(e)}")