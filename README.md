
# Sentiment Analysis for Twitter data


This repository contains a simple implementation of sentiment analysis on Twitter data using a Decision Tree Classifier. The goal is to classify tweets into three categories: hate speech, offensive language, and no hate or offensive language.

## Dataset

The dataset used in this project (twitter.csv) is a collection of tweets along with their corresponding labels (classes). The labels are mapped to three categories: 0 for hate speech, 1 for offensive language, and 2 for tweets with no hate or offensive language. The Pandas library is employed to read and manipulate the dataset.
    




run

```bash
  import pandas as pd
  import numpy as np

  dataset = pd.read_csv("twitter.csv")
  dataset['labels'] = dataset['class'].map({
      0: "hate speech",
      1: "offensive language",
      2: "no hate or offensive language"
  })
  data = dataset[["tweet", "labels"]]


```


## Data Cleaning

The tweets undergo a thorough preprocessing phase to ensure uniformity and to remove unnecessary elements that may not contribute to the sentiment analysis. Techniques such as converting text to lowercase, removing URLs, special characters, punctuation, and stopwords, as well as stemming, are employed to clean the textual data.
```bash
  import re
  import nltk
  from nltk.corpus import stopwords
  from nltk.stem import SnowballStemmer
  import string

  stopword = set(stopwords.words("english"))
  stemmer = SnowballStemmer("english")

  def clean(text):
      # ... (code for data cleaning)
      return text

  data["tweet"] = data["tweet"].apply(clean)


```

## Feature Extraction

Text data is converted into a numerical format using the CountVectorizer from scikit-learn. This step is crucial for machine learning models to understand and process the textual information.

```bash
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.model_selection import train_test_split

  cv = CountVectorizer()
  x = cv.fit_transform(np.array(data["tweet"]))
  y = np.array(data["labels"])

```
## Model Training
A Decision Tree Classifier is chosen as the machine learning model for sentiment analysis. The dataset is split into training and testing sets, and the model is trained on the training set.

```bash
  from sklearn.tree import DecisionTreeClassifier

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  dt = DecisionTreeClassifier()
  dt.fit(x_train, y_train)
```
## Model Evaluation
The model is evaluated using a confusion matrix and accuracy score. The confusion matrix provides insights into the model's performance for each class, while the accuracy score indicates the overall accuracy of the model on the test set.
```bash
  from sklearn.metrics import confusion_matrix, accuracy_score
  import seaborn as sns
  import matplotlib.pyplot as plt

  y_pred = dt.predict(x_test)
  cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
  sns.heatmap(cm, annot=True, cmap='plasma')
  plt.show()

# Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}")
```
## Inference

The trained model can be utilized to predict the sentiment of new tweets. A sample tweet is preprocessed using the same cleaning techniques, converted into a numerical format, and fed into the model for prediction.
```bash
  sample = "lets unite and kill all the people who are all protesting against the government"
  sample = clean(sample)
  data1 = cv.transform([sample]).toarray()
  prediction = dt.predict(data1)

  print(f"Prediction: {prediction[0]}")
```
