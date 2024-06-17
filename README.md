# data_exploration
Inspired by https://labs.google.com/code/dsa?view=readme, this script call some LLMs to do some basic data science analysis of a CSV file. In the end all the executed cells will be converted to a Jupyter notebook. The script uses Jupyter client to run Python code interactively and relies on OpenAI APIs to make LLM calls.

Execution loop is: 1. generate a plan; 2. figure out which step we're in with historical information; 3. generate code and execute, add the states in history. Step 2 and 3 are repeated until MAX_STEPS (hardcoded at 10) are reached.