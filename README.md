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

Comment Categorization & Reply Assistant Tool

This repository contains a Python project utilizing a Logistic Regression classifier trained with TF-IDF features to categorize user comments into predefined labels (e.g., Praise, Hate/Abuse, Constructive Criticism, Threat). It then suggests an automated, appropriate response for the detected category using a Gradio web interface.

# How to Run the Code

This project is designed to be run within a Jupyter Notebook environment, such as Google Colab, due to its use of the files.upload() function for dataset loading and Gradio for the UI.

Prerequisites

Python: Ensure you have Python 3.x installed (if running locally).

Dataset: You need a CSV file for training the model. The file must have at least two columns:

comment (containing the text data)

label (containing the category labels)

Installation

The project relies on several common Python libraries. You can install all required packages using pip.

pip install gradio pandas scikit-learn matplotlib


Execution (Recommended: Google Colab)

The easiest way to run the code and see the interactive UI is in Google Colab.

Open the Notebook: Upload and open the Comment_Categorization.ipynb file in Google Colab.

Run Cells in Sequence: Execute each code cell one by one:

Cell 1 (Installation): Runs the !pip install command to install necessary libraries.

Cell 2 (Imports & Data Upload): This cell uses the from google.colab import files; files.upload() command. When you run this cell, a file selection widget will appear. You must upload your CSV dataset here (e.g., comment_dataset_1000_varied.csv).

Cells 3, 4, 5 (Data Prep & Training): These cells read the data, handle duplicates, and then train the Logistic Regression model. Run them to prepare the classifier.

Final Cell (Launch Gradio UI): This cell defines the Gradio interface and calls iface.launch().

Upon execution, the output will provide a public URL (e.g., https://c639d0f31577b6d817.gradio.live) which hosts the interactive web application.

Execution (Local Environment)

If you run the notebook on your local machine using a Jupyter Notebook server, you will need to manually change the data loading step.

Installation: Run the pip install command above.

Modify Data Loading: The original notebook uses:

from google.colab import files
uploaded = files.upload() 
df = pd.read_csv(list(uploaded.keys())[0])


Replace this with a local path to your CSV file:

# Load dataset from a local path
df = pd.read_csv('path/to/your/comment_data.csv') 


Run Remaining Cells: Execute the remaining cells to train the model and launch the Gradio interface. Gradio will typically launch on http://127.0.0.1:7860/.

# User Interface (Gradio)

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

# Sample Results

<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/48759be8-82c7-4d7d-9d7a-1dbfa73db983" />
<img width="1920" height="906" alt="image" src="https://github.com/user-attachments/assets/ae393c1f-aaaa-4f33-b896-68daeb362243" />
<img width="1920" height="902" alt="image" src="https://github.com/user-attachments/assets/da16e151-b02d-4bda-9468-8dea6602625d" />
<img width="1920" height="901" alt="image" src="https://github.com/user-attachments/assets/c7bbb842-6a39-4b3d-934b-15060343bf8d" />


# Future Improvements

- The project currently uses a basic Logistic Regression model with TF-IDF features. Here are several opportunities for future development:

- Expand Category Scope: Introduce more nuanced categories like "Feature Request," "Bug Report," and "Off-Topic Discussion."

- Multilingual Support: Integrate a language detection step and train models capable of categorizing comments in multiple languages (e.g., Spanish, French) to broaden the tool's applicability.

- Advanced Model Architecture: Migrate from Logistic Regression to deep learning models (e.g., using a pre-trained transformer model like BERT or RoBERTa) for potentially higher accuracy and better handling of context.

- Deployment Optimization: Optimize the model size and deployment package to allow for faster inference times and easier deployment to platforms like Hugging Face Spaces or a serverless function.

- Feedback Loop Integration: Implement a way for users to provide feedback on classification errors, allowing the model to be continuously re-trained and improved based on real-world use.

- Sentiment and Urgency Scoring: Add an additional output layer to the model to provide a separate sentiment score (positive/negative) and an urgency score, which is critical for prioritizing replies to threats or bug reports.





