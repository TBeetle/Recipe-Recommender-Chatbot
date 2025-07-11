"""
Recipe Recommender Chatbot Preprocessing Module

This module provides functions to preprocess recipe data, build regex patterns for intent recognition,
and filter recipes based on user intents. It is designed to be used in the recipe recommender chatbot application.

Contributor(s):
    Tyler Beetle <twbeetle@outlook.com>

Change Log
----------
2025-07-10 :: Refactored to improve logging, error handling, and modularity.
"""

import re
import ast

def preprocess_data(df):
    '''
    Preprocesses the input DataFrame by converting string representations of lists to actual lists

    Parameters
    ----------
    df  : pd.DataFrame
        Input DataFrame containing recipe data

    Returns
    -------
    df  : pd.DataFrame
        Processed DataFrame with converted list columns
    '''

    required_columns = ['tags', 'ingredients']

    # Check if required columns exist in the DataFrame
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Loop through each required column
    for col in required_columns:
        # Convert string representations of lists to actual lists
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
        )

    return df

def build_intent_patterns(df):
    '''
    Builds regex patterns for intent recognition based on recipe tags and ingredients

    Parameters
    ----------
    df  : pd.DataFrame
        Preprocessed DataFrame with recipe data

    Returns
    -------
    patterns  : dict
        Dictionary containing regex patterns for different intent categories
    '''

    # Define regex patterns for different intent categories
    patterns = {
        'dietary': {},
        'cuisine': {},
        'meal_type': {},
        'time_constraint': {
            'quick': r'\b(quick|fast|easy|30.min|minutes|rushed)\b',
            'slow': r'\b(slow|hours|all.day|weekend|complex)\b'
        },
        'ingredients': {},
        'special': {
            'greeting': r'\b(hi|hello|hey)\b',
            'goodbye': r'\b(bye|goodbye|exit)\b',
            'help': r'\b(help|what can you do|commands|options)\b'
        }
    }

    # Define keywords for dietary, cuisine, and meal type intents
    DIETARY_KEYWORDS = [
        'vegetarian', 'vegan', 'healthy', 'low-fat', 'gluten', 'diet', 'low-protein',
        'low-sodium', 'low-cholesterol', 'low-saturated-fat', 'low-calorie',
        'free-of-something', 'low-in-something', 'very-low-carbs', 'egg-free',
        'lactose', 'diabetic', 'dairy-free'
    ]
    CUISINE_KEYWORDS = [
        'asian', 'italian', 'mexican', 'indian', 'american', 'chinese', 'japanese',
        'thai', 'korean', 'mediterranean', 'north-american', 'southwestern-united-states',
        'brazilian', 'south-american', 'european', 'french', 'greek', 'spanish',
        'portuguese', 'german', 'hungarian', 'austrian', 'dutch', 'australian',
        'canadian', 'english', 'middle-eastern', 'turkish', 'african', 'moroccan',
        'russian', 'colombian', 'filipino', 'scandinavian', 'vietnamese', 'lebanese',
        'swiss', 'cajun', 'creole', 'irish', 'polish', 'swedish', 'norwegian',
        'danish', 'egyptian', 'ethiopian', 'nigerian', 'belgian', 'venezuelan',
        'quebec', 'british-columbian', 'nepalese', 'puerto-rican', 'costa-rican',
        'guatemalan', 'honduran', 'argentine', 'chilean', 'iranian-persian',
        'libyan', 'somalian', 'beijing', 'mongolian', 'cambodian', 'sudanese'
    ]
    MEAL_TYPE_KEYWORDS = [
        'desserts', 'main', 'main-dish', 'appetizer', 'beverage', 'breakfast',
        'lunch', 'dinner', 'snack', 'starter', 'entree', 'side-dishes',
        'pies-and-tarts', 'pies', 'cakes', 'cookies-and-brownies', 'biscotti',
        'soups-stews', 'stews', 'salads', 'sandwiches', 'casseroles', 'muffins',
        'breads', 'quick-breads', 'rolls-biscuits', 'cupcakes', 'pasta',
        'pasta-rice-and-grains', 'rice', 'lasagna', 'spaghetti',
        'omelets-and-frittatas', 'pancakes-and-waffles', 'pizza', 'potluck',
        'potatoes', 'scones', 'sauces', 'salsas', 'dips', 'spread', 'snacks',
        'smoothies', 'burgers', 'veggie-burgers', 'salad-dressings', 'candy',
        'fudge', 'bar-cookies', 'brownies', 'puddings-and-mousses',
        'cobblers-and-crisps'
    ]

    tag_set, ingredient_set = set(), set()

    # Collect unique tags from the DataFrame
    for tags in df['tags']:
        if isinstance(tags, list):
            tag_set.update(t.lower() for t in tags)
    
    # Collect unique ingredients from the DataFrame
    for ings in df['ingredients']:
        if isinstance(ings, list):
            ingredient_set.update(i.lower() for i in ings)

    # Build regex patterns for dietary, cuisine, and meal type intents
    for tag in tag_set:
        if any(x in tag for x in DIETARY_KEYWORDS):
            patterns['dietary'][tag] = rf'\b({re.escape(tag)})\b'

        elif any(x in tag for x in CUISINE_KEYWORDS):

            patterns['cuisine'][tag] = rf'\b({re.escape(tag)})\b'
        elif any(x in tag for x in MEAL_TYPE_KEYWORDS):

            patterns['meal_type'][tag] = rf'\b({re.escape(tag)})\b'
    
    # Build regex patterns for ingredients
    for ing in ingredient_set:
        if len(ing) > 2:
            patterns['ingredients'][ing] = rf'\b({re.escape(ing)})\b'

    return patterns