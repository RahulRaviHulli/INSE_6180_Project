# Sentiment Analysis on Reddit Comments and Posts to detect Depression
Using sentiment analysis on Reddit comments and posts can provide valuable insights into detecting signs of depression within the community, potentially aiding in early intervention and support. By analyzing the sentiment expressed in these interactions, patterns indicative of depression can be identified, enabling targeted outreach and mental health resources allocation. Leveraging machine learning algorithms, sentiment analysis offers a scalable approach to monitor the emotional well-being of Reddit users and facilitate timely interventions when needed.

## Below is an explanation of the project at depth
### 1: Setting the Environment
#### 1.1 Mounting Google Drive

These three lines of code are used to mount Google Drive in Google Colab, which allows you to access files and directories stored in your Google Drive from within the Colab environment. Here's a brief explanation of each line:

from google.colab import drive: This line imports the drive module from the google.colab package. This module provides functions to interact with Google Drive.

drive.mount('/content/drive'): This line calls the mount function of the drive module and mounts your Google Drive at the specified directory /content/drive within the Colab environment. After running this line and following the authentication instructions, you'll be able to access your Google Drive files under this directory.

These lines are commonly used at the beginning of Colab notebooks to set up access to Google Drive for loading and saving files.

#### 1.2 Setting a folder as a working directory

The two lines provided are used to navigate to a specific directory within your Google Drive and list its contents. Here's what each line does:

%cd /content/drive/My Drive/Sentiment Analysis Using LSTM: This line changes the current working directory (%cd) to /content/drive/My Drive/Sentiment Analysis Using LSTM. It's using a magic command %cd to change the directory.

!ls: This line executes a shell command (!) to list (ls) the contents of the current directory.

Together, these lines navigate to the specified directory in your Google Drive and list its contents, allowing you to see what files and subdirectories are present there. This is helpful for organizing your work and ensuring you're accessing the correct files.

#### 1.3 Install Requirements

The lines !pip install tensorflow, !pip install keras, and !pip install keras_preprocessing are used to install the required Python packages (tensorflow, keras, and keras_preprocessing, respectively) in your Google Colab environment.

Why do I run this everytime I run the code? it's due to the fact that Google Colab provides a temporary runtime environment. Each time you start a new session or reopen a notebook, the environment is reset, and any previously installed packages are not persisted. Therefore, you need to reinstall the required packages to ensure they are available for use in your current session.

#### 1.4 Import Environment

pandas as pd: Used for data manipulation and analysis.
numpy as np: Provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.
re: Allows for regular expression operations.
string: Provides a collection of string constants and helper functions.
nltk: Natural Language Toolkit library for natural language processing tasks.
stopwords: A list of common stopwords in various languages.
PorterStemmer: An algorithm for reducing words to their word stems or roots.
tensorflow as tf: Machine learning framework for building and training models.
layers: Module within TensorFlow providing various types of layers for building neural networks.
one_hot, Tokenizer, Sequential, pad_sequences, Embedding, Conv1D, LSTM: Various modules and classes from Keras for text preprocessing and building neural network models.
tokenizer_from_json: Utility function for loading a tokenizer from JSON format.
seaborn as sns: Statistical data visualization library based on matplotlib.
matplotlib.pyplot as plt: Module providing a MATLAB-like interface for plotting.
train_test_split: Function for splitting datasets into training and testing subsets.
confusion_matrix, classification_report: Functions for evaluating model performance.

#### 1.5 Download Stopwords

The purpose of the code snippet nltk.download("stopwords") is to download the stopwords corpus from the NLTK (Natural Language Toolkit) library. Stopwords are common words like "the," "is," "and," etc., that are often removed from text data during natural language processing tasks because they typically do not provide much information about the content of the text.

After downloading the stopwords corpus, the code initializes a SnowballStemmer object from NLTK, which is used for stemming. Stemming is the process of reducing words to their root form by removing suffixes and prefixes. This process helps in reducing the dimensionality of the text data and improving the performance of natural language processing tasks.

Finally, the code creates a set called stopword containing English stopwords obtained from the NLTK stopwords corpus. This set is then used later in the text preprocessing steps to remove stopwords from text data.

#### 1.6 Importing the Reddit Comments Dataset

The code data = pd.read_csv("depression_dataset_reddit_cleaned.csv") reads a CSV file named "depression_dataset_reddit_cleaned.csv" into a pandas DataFrame called data. This dataset likely contains cleaned Reddit comments related to depression. By using the read_csv function from the pandas library, the data is loaded into memory, allowing for further analysis and processing.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 2: Working on the Dataset

#### 2.1 Dataset Exploration

The code snippet data.shape displays the dimensions of the DataFrame data, indicating the number of rows and columns in the dataset.

The code data.head(5) displays the first five rows of the DataFrame data, providing a quick overview of the data's structure and content. This allows users to inspect the data and understand its format before further analysis.

#### 2.2 Are there any Null Values?

The code data.isnull().values.any() checks whether there are any null values present in the DataFrame data. If there are any null values, the function will return True; otherwise, it will return False. This helps in identifying if there are missing values in the dataset that need to be handled before further analysis.

#### 2.3 Data Shape

The code data.value_counts().sum() attempts to count the total number of occurrences of each unique value across all columns in the DataFrame data. This operation will not yield the total number of values in the DataFrame but instead counts occurrences separately for each column.

#### 2.4 Measure Classes

The code data['is_depression'].value_counts() calculates the frequency of each class label (depression and non-depression) in the 'is_depression' column of the DataFrame data. Following this, sns.countplot(x='is_depression', data=data) creates a count plot to visualize the distribution of classes, where 'is_depression' is plotted on the x-axis, showing the count of each class.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 3: Data Preprocessing

#### 3.1 What does the comments look like?

This code retrieves the contents of the "clean_text" column for the third row (index 2) of the DataFrame data. It displays the text of the comment at index 2.

#### 3.2 Preprocessing Function

This code defines a function named clean for text preprocessing. It performs the following operations:

- Converts the text to lowercase.
- Removes square brackets and their contents using a regular expression.
- Removes URLs using a regular expression.
- Removes HTML tags using a regular expression.
- Removes punctuations using the string.punctuation set.
- Removes newline characters.
- Removes alphanumeric characters that contain digits.
- Applies stemming using the Porter Stemmer algorithm to reduce words to their root form. Finally, it returns the preprocessed text.

#### 3.3 Text Preprocessing: Cleaning the Text Data

This code segment applies the clean function defined earlier to the "clean_text" column of the dataset data. It preprocesses the text data by cleaning it. After preprocessing, it designates the preprocessed text as x and the corresponding target variable "is_depression" as y. This prepares the data for further processing and model training.

#### 3.4 Traing and Test Split

This code segment applies the clean function defined earlier to the "clean_text" column of the dataset data. It preprocesses the text data by cleaning it. After preprocessing, it designates the preprocessed text as x and the corresponding target variable "is_depression" as y. This prepares the data for further processing and model training.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 4: Text Vectorization

#### 4.1 Text Vectorization Configuration

This code sets up the configuration for text vectorization using TensorFlow's TextVectorization layer. It specifies parameters such as max_tokens to limit the vocabulary size, output_mode to output integer sequences, and output_sequence_length to set the maximum length of output sequences. The adapt method then adapts the text vectorizer to the training data (X_train). This process helps tokenize and vectorize the text data for input into a neural network model.

#### 4.2 Valuate top and bottom 5 vocab size

This code snippet evaluates and prints the top 5 most common words and the bottom 5 least common words in the vocabulary generated by the TextVectorization layer. It first retrieves the vocabulary using the get_vocabulary() method. Then, it slices the vocabulary list to obtain the top 5 and bottom 5 words. Finally, it prints the total vocabulary size along with the top and bottom words.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

