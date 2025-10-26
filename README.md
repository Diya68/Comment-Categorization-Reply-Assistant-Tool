# Comment-Categorization-Reply-Assistant-Tool
Leveraging NLP, this project categorizes user comments (praise, hate, criticism, spam, etc.) to help teams respond efficiently. Features include ML-powered analysis, flexible input/output, and optional UI/visualizations. Python-based with scikit-learn, pandas, NLTK/spaCy, and HuggingFace Transformers. Ideal for enhancing feedback management.

# Objective 
The objective of this project is to combine Natural Language Processing (NLP) and a machine learning model trained on a proprietary dataset to categorize user comments into predetermined groups. Real-world conversational contexts are represented by the manually gathered and categorized comments in the dataset. Following classification, an automatically generated suggested response that fits the anticipated category is also provided by the system. For applications like automated help, social engagement, and content moderation, the project incorporates an intuitive Gradio interface that allows users to enter comments and receive results instantaneously. 

# Problem Statement 
Users post a wide variety of comments. These could be:
- Appreciative (praise/support)
- Emotional
- Abusive (hate/threat)
- Constructively negative (e.g., I didn't like the design but appreciate the effort)
- Spam or irrelevant
- Questions or suggestions

The tool must sort comments into buckets so that the team can:
- Engage positively
- Address genuine criticism
- Ignore spam
- Escalate threats/hate
- Provide answers where needed

- # Tech Stacks
- Language: Python
- Libraries: scikit-learn, pandas, Matplotlib
- UI: Gradio
- Model: Logistic Regression, SVM

# How to Run the Code
This project is designed to be run within a Jupyter Notebook environment, such as Google Colab, due to its use of the files.upload() function for dataset loading and Gradio for the UI

- Prerequisites
Python: Ensure you have Python 3.x installed (if running locally)
Dataset: You need a CSV file for training the model. The notebook is configured to look for a file after being uploaded. The file must have at least two columns:
comment (containing the text data)
label (containing the category labels)

- Installation
The project relies on several common Python libraries. You can install all required packages using pip.
pip install gradio pandas scikit-learn matplotlib

- Execution (Recommended: Google Colab)
The easiest way to run the code and see the interactive UI is in Google Colab.
Open the Notebook: Upload and open the Comment_Categorization.ipynb file in Google Colab.
Run Cells in Sequence: Execute each code cell one by one:
Cell 1 (Installation): Runs the !pip install command to install necessary libraries.
Cell 2 (Imports & Data Upload): This cell uses the from google.colab import files; files.upload() command. When you run this cell, a file selection widget will appear. You must upload your CSV dataset here (e.g., comment_dataset_1000_varied.csv).
Cells 3, 4, 5 (Data Prep & Training): These cells read the data, handle duplicates, and then train the Logistic Regression model. Run them to prepare the classifier.
Final Cell (Launch Gradio UI): This cell defines the Gradio interface and calls iface.launch().
Upon execution, the output will provide a public URL (e.g., https://c639d0f31577b6d817.gradio.live) which hosts the interactive web application.

- Execution (Local Environment)
If you run the notebook on your local machine using a Jupyter Notebook server, you will need to manually change the data loading step.

- Installation: Run the pip install command above.
Modify Data Loading: The original notebook uses:
from google.colab import files
uploaded = files.upload() 
df = pd.read_csv(list(uploaded.keys())[0])

- Load dataset from a local path
df = pd.read_csv('path/to/your/comment_data.csv') 
Run Remaining Cells: Execute the remaining cells to train the model and launch the Gradio interface. Gradio will typically launch on http://127.0.0.1:7860/.

- User Interface (Gradio)
The final cell launches a web UI built with Gradio. The interface consists of two main components:
Component
Function
Your Comment (Input Textbox)
Where you enter the user comment you want to categorize.
Predicted Category (HTML Output)
Displays the predicted category (e.g., Praise, Threat) with color coding for quick identification.
Suggested Reply (Output Textbox)
Provides a pre-written, appropriate reply based on the predicted category.
The UI title is 🐼 Comment Categorization & Reply Assistant 🐼.

