# data_exploration
Inspired by https://labs.google.com/code/dsa?view=readme, this script call some LLMs to do some basic data science analysis of a CSV file. In the end all the executed cells will be converted to a Jupyter notebook. The script uses Jupyter client to run Python code interactively and relies on OpenAI APIs to make LLM calls.

Execution loop is: 1. generate a plan; 2. figure out which step we're in with historical information; 3. generate code and execute, add the states in history. Step 2 and 3 are repeated until MAX_STEPS (hardcoded at 10) are reached.

To run the script,
1. Put a CSV file at project folder
2. In a virtual env, `pip install -r requirements.txt`
3. `OPENAI_API_KEY=<API_KEY> MODEL=<OpenAI model name, default is gpt-3.5-turbo> FILE=<CSV file name, default is bird_migration.csv> python data_exploration/main.py`