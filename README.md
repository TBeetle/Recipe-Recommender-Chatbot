# Recipe Recommender Chatbot

A chatbot that recommends recipes based on user queries. The model can pick up on dietary preferences, cuisines, meal types, time constraints, and ingredients. The system uses semantic intent recognition and natural language processing to understand both keyword and conversational queries.

## Features

- Conversational recipe recommendations (e.g., “Give me Asian chicken recipes please”)
- Recognizes dietary, cuisine, meal type, time constraints, and ingredient intents
- Handles natural language and filters out filler words/phrases
- Returns top matching recipes with cooking time, tags, ingredients, and a description

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd recipe_chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the data:**
   - Ensure `CLEANED_recipes.csv` is present in the `data/` directory.
   - If not, place your cleaned recipe CSV file as `data/CLEANED_recipes.csv`.

## Usage

Run the chatbot:

```bash
python chatbot.py
```

You’ll see:
```
🍳 RECIPE RECOMMENDER CHATBOT 🍳
Type 'quit' to exit.
You:
```

Type your query, for example:
- `Give me pasta recipes with beef`
- `Show me quick vegan desserts`
- `I want Italian chicken recipes please`

The chatbot will recommend up to 3 matching recipes.

## Project Structure

```
recipe_chatbot/
  chatbot.py                # Main chatbot script
  requirements.txt          # Python dependencies
  README.md                 # Project overview
  data/
    CLEANED_recipes.csv     # Cleaned recipe dataset
  utils/
    preprocess.py           # Data preprocessing and intent pattern building
    model.py                # Model and embedding utilities
    intent.py               # Intent recognition logic
    filter.py               # Recipe filtering and formatting
```

## Customization

- **Dataset:** Replace `data/CLEANED_recipes.csv` with your own recipe data (ensure it has `tags` and `ingredients` columns as lists).
- **Intent Patterns:** Update `utils/preprocess.py` if you want to expand recognized cuisines, meal types, or dietary tags.
- **Filler Words:** Edit the `FILLER_PHRASES` list in `utils/intent.py` to fine-tune conversational understanding.

## Logging

- All conversations and errors are logged to `conversation.log` with rotation to prevent file bloat.

## License

MIT License (or your preferred license) 