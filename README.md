# Comment Toxicity Detector 💬🚫

The Comment Toxicity Detector is a machine learning project aimed at identifying toxic comments in text. It uses natural language processing (NLP) techniques to classify comments as toxic or non-toxic, helping to automate the moderation of online communities.

# 🚀 Key Features

	•	Toxicity Classification: Detects toxic language in comments, including categories like insult, threat, and hate speech.
	•	End-to-End NLP Pipeline: From data preprocessing to model training and prediction.
	•	Pretrained Model: Efficiently classifies comments using a trained model.
	•	Dataset: Based on the Jigsaw Toxic Comment Classification Challenge, which contains a large set of labeled comments.
    link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

# 🧠 How It Works

The project uses the Jigsaw Toxic Comment Classification Challenge dataset. The model is designed to:

	1.	Data Preprocessing:
	•	Tokenizes and vectorizes comments for training and testing.
	•	Cleans text to remove unnecessary characters and stopwords.
	2.	Model Training:
	•	The model is trained using multiple labels to classify comments as toxic, severe toxic, obscene, threat, insult, or identity hate.
	3.	Prediction:
	•	After training, the model predicts the toxicity of new comments and outputs the corresponding class labels.

# 🔧 Usage

	1.	Jupyter Notebook:
	•	Run the CommentToxicity.ipynb to load the dataset, train the model, and classify new comments.
