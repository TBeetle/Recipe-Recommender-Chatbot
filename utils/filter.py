"""
Recipe Recommender Chatbot Filtering Module

This module provides functions to filter recipes based on recognized intents and user input.
It processes the recipe DataFrame to return relevant recipes based on dietary preferences, cuisines,
meal types, time constraints, and ingredients.

Contributor(s):
    Tyler Beetle <twbeetle@outlook.com>

Change Log
----------
2025-07-10 :: Refactored to improve logging, error handling, and modularity.
"""
import re
import logging

def filter_recipes(intents, user_input, df):
    '''
    Filters recipes based on recognized intents and user input

    Parameters
    ----------
    intents  : dict
        Recognized intents from user input
    user_input  : str
        User's input text
    df  : pd.DataFrame
        Recipe DataFrame

    Returns
    -------
    df  : pd.DataFrame
        Filtered DataFrame with matching recipes
    '''

    # Ensure the DataFrame is a copy
    df = df.copy()

    def normalize(col: str) -> str:
        return col.strip().lower()
    
    # Normalize column names to handle case and whitespace
    col_map = {normalize(col): col for col in df.columns}
    logging.debug(f"Original DataFrame columns: {list(df.columns)}")

    # Rename columns to standard names if they exist in the DataFrame
    if 'ingredients' not in df.columns and 'ingredients' in col_map:
        logging.warning(f"Renaming column '{col_map['ingredients']}' to 'ingredients'")
        df.rename(columns={col_map['ingredients']: 'ingredients'}, inplace=True)
    
    # If 'tags' column is missing, check for alternative names
    if 'tags' not in df.columns and 'tags' in col_map:
        logging.warning(f"Renaming column '{col_map['tags']}' to 'tags'")
        df.rename(columns={col_map['tags']: 'tags'}, inplace=True)

    user_input = user_input or ""
    user_words = set(re.findall(r'\w+', user_input.lower()))
    tag_filters = []
    ingredient_filters = []

    # Extract tags and ingredients from intents
    for col, values in intents.items():
        if col in ['dietary', 'cuisine', 'meal_type'] and values:
            tag_filters.extend(v.lower() for v in values)
        elif col == 'ingredients' and values:
            ingredient_filters.extend(v for v in values if v.lower() in user_words)

    logging.debug(f"Applying tag filters: {tag_filters}, ingredient filters: {ingredient_filters}")

    # Filter recipes based on tags
    if tag_filters:
        before_count = len(df)
        df = df[df['tags'].apply(
            lambda x: isinstance(x, list) and any(t.lower() in tag_filters for t in x)
        )]
        logging.debug(f"After tag filters: {len(df)} recipes (filtered {before_count - len(df)})")

    # If no tag filters are applied, check for cuisine keywords in user input
    if not tag_filters and user_words:
        cuisine_keywords = {
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
        }
        user_cuisines = user_words.intersection(cuisine_keywords)
        if user_cuisines:
            before_count = len(df)
            df = df[df['tags'].apply(
                lambda x: isinstance(x, list) and any(cuisine in str(t).lower() for t in x for cuisine in user_cuisines)
            )]
            logging.debug(f"After fallback cuisine filter {user_cuisines}: {len(df)} recipes (filtered {before_count - len(df)})")
    
    # Filter recipes based on ingredients
    for ing in ingredient_filters:
        before_count = len(df)
        df = df[df['ingredients'].apply(
            lambda x: isinstance(x, list) and any(ing in str(i).lower() for i in x)
        )]
        logging.debug(f"After ingredient filter '{ing}': {len(df)} recipes (filtered {before_count - len(df)})")
    
    # If no ingredients were specified, check for ingredients in user input
    if 'time_constraint' in intents and intents['time_constraint']:
        values = intents['time_constraint']

        # Apply time constraints based on user input
        if 'quick' in values:
            before_count = len(df)
            df = df[df['minutes'] <= 60]
            logging.debug(f"After 'quick' time filter: {len(df)} recipes (filtered {before_count - len(df)})")
        elif 'slow' in values:
            before_count = len(df)
            df = df[df['minutes'] > 120]
            logging.debug(f"After 'slow' time filter: {len(df)} recipes (filtered {before_count - len(df)})")

    logging.info(f"Filtered recipes count: {len(df)}")
    return df

def format_recipe(recipe_row, index):
    '''
    Formats a single recipe into a readable string

    Parameters
    ----------
    recipe_row  : pd.Series
        Single row of recipe data
    index  : int
        Recipe index for display

    Returns
    -------
    recipe_str  : str
        Formatted recipe string
    '''
    name = str(recipe_row.get('name', 'Unknown')).title()
    minutes = recipe_row.get('minutes', 0)

    # Try to convert minutes to an integer, defaulting to 0 if conversion fails
    try:
        minutes = int(float(minutes))
    except Exception:
        minutes = 0

    # Format time as "X minutes" or "Xh Ym"
    time = f"{minutes} minutes" if minutes <  60 else f"{minutes // 60}h {minutes % 60}m"

    # Extract ingredients and tags, ensuring they are lists
    ingredients = recipe_row.get('ingredients', []) if isinstance(recipe_row.get('ingredients'), list) else []
    ingredients_str = ", ".join(ingredients[:5]) + (
        f" (and {len(ingredients) - 5} more)" if len(ingredients) > 5 else ""
    )

    # Extract tags, ensuring they are lists and formatting them`
    tags = recipe_row.get('tags', []) if isinstance(recipe_row.get('tags'), list) else []
    tags_str = " • ".join(tags[:4]) if tags else "No tags available"

    # Ensure description is a string and handle cases where it might be NaN or empty
    description = recipe_row.get('description', "No description available")
    if not isinstance(description, str) or description.lower() == 'nan':
        description = "No description available"

    return (
        f"\n🍽️  {index}. {name}\n"
        f"⏰ Cooking Time: {time}\n"
        f"🏷️  Tags: {tags_str}\n"
        f"🥘 Main Ingredients: {ingredients_str}\n"
        f"📝 Description: {description}\n"
        f"{'─' * 50}"
    )