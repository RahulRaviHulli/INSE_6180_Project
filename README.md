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

### 5: Create Embedding Layer

#### 5.1 Standard Embedding Layer Configuration

This code defines an embedding layer configuration using the Embedding class from Keras. Here's what each parameter signifies:

- input_dim: The size of the vocabulary, which is the maximum number of tokens that can be represented by the embedding layer.
- output_dim: The dimensionality of the embedding space. It determines the size of the vector representation for each token.
- embeddings_initializer: The initialization strategy for the embedding weights. In this case, it's set to "uniform", meaning the weights are initialized using a uniform distribution.
- input_length: The length of input sequences that will be fed into the embedding layer. This parameter is required if you plan to connect a Flatten or Dense layer downstream, but it's not necessary for LSTM or Conv1D layers.
Overall, this configuration sets up an embedding layer suitable for tokenizing text data with a vocabulary size of max_vocab_length, producing dense embeddings of dimensionality 128, and input sequences of length max_length.

#### 5.2 Model Definition: Dense Architecture

This code defines a dense neural network architecture for text classification. Here's a breakdown of what each part does:

- inputs: Defines the input layer for the model, specifying the shape and data type of the input data. In this case, it expects input sequences of strings.
- text_vectorizer(inputs): This is the text vectorization layer previously configured. It converts raw text inputs into sequences of integers.
- embedding(x): This line applies the embedding layer to the integer sequences obtained from the text vectorization. It converts the integer-encoded tokens into dense vectors of fixed size.
- layers.GlobalAveragePooling1D(): This layer performs global average pooling over the sequence dimension. It reduces the dimensionality of the input by taking the average of all values across the sequence dimension.
- layers.Dense(1, activation="sigmoid"): This is the output layer of the model. It consists of a single neuron with a sigmoid activation function, which is commonly used for binary classification tasks.
- model_1: This line constructs the Keras Model object, specifying the inputs and outputs of the model. The model is named "model_1_dense".
Overall, this architecture takes string inputs, converts them to integer sequences, applies an embedding layer to obtain dense representations, performs global average pooling to reduce dimensionality, and finally, passes the result through a dense layer with a sigmoid activation function for binary classification.

#### 5.3 Model Compilation

This code compiles the defined model (model_1) with the specified loss function, optimizer, and evaluation metrics. Here's what each argument does:

- loss='binary_crossentropy': This specifies the loss function to use during training. Binary crossentropy is commonly used for binary classification problems like this one.

- optimizer='adam': This specifies the optimizer to use for training the model. Adam is a popular choice due to its adaptive learning rate properties and efficiency in training neural networks.

- metrics=['accuracy']: This specifies the evaluation metric(s) to monitor during training and testing. In this case, it uses accuracy, which measures the proportion of correct predictions made by the model.

With this compilation step, the model is configured for training with the specified loss function, optimizer, and evaluation metric.

#### 5.4 General Model Training

This code trains the compiled model (model_1) using the training data (X_train, Y_train) and validates it on the validation data (X_test, Y_test). Here's what each argument does:

- X_train, Y_train: These are the input features and corresponding target labels for training the model.

- validation_data=(X_test, Y_test): This specifies the validation data to evaluate the model's performance after each epoch. It helps monitor whether the model is overfitting or generalizing well to unseen data.

- epochs=5: This parameter determines the number of training epochs, i.e., the number of times the model will be trained on the entire training dataset. One epoch is a single forward and backward pass of all the training examples.

During training, the model's weights are adjusted iteratively to minimize the specified loss function (binary crossentropy in this case) using the optimizer (Adam) based on the training data. The validation data is used to monitor the model's performance on unseen data and prevent overfitting.

#### 5.5 General Classification

The provided code is for making predictions using a trained model (model_1) on the test data (X_test). It then converts the predicted probabilities into binary predictions by thresholding at 0.5, assigning values of 1 for probabilities greater than or equal to 0.5 and 0 otherwise.

Here's a breakdown of the code:

- Y_pred = model_1.predict(X_test): This line uses the trained model model_1 to predict the labels for the test data X_test.

- Y_pred = (Y_pred >= 0.5).astype("int"): This line converts the predicted probabilities (Y_pred) into binary predictions. It assigns 1 to elements where the predicted probability is greater than or equal to 0.5, and 0 otherwise.

The final line is not explicitly shown but it's implied that Y_pred contains the binary predictions for the test data, which can then be used for evaluation or further analysis.

#### 5.6 Accuracy, Macro Average, Weighted Average

The provided code calculates and prints a classification report, including metrics such as accuracy, macro average, and weighted average, based on the true labels (Y_test) and the predicted labels (Y_pred). This report provides a comprehensive overview of the model's performance across different classes.

Here's what each line does:

- print(classification_report(Y_test, Y_pred)): This line generates a classification report using the true labels (Y_test) and the predicted labels (Y_pred). The report includes metrics such as precision, recall, F1-score, and support for each class, as well as overall accuracy, macro average, and weighted average.
The classification report provides valuable insights into the model's performance, helping to assess its effectiveness in classifying instances from each class and overall.

#### 5.7 Using Logistic Regression for Classification

- TF-IDF Vectorizer Definition: It initializes a TF-IDF vectorizer (TfidfVectorizer) with a specified maximum number of features (max_features). This vectorizer converts text data into numerical features based on the TF-IDF (Term Frequency-Inverse Document Frequency) representation.

- Logistic Regression Classifier Definition: It initializes a logistic regression classifier (LogisticRegression). This classifier will be used for binary classification tasks.

- Pipeline Creation: It creates a pipeline using the Pipeline class from scikit-learn. The pipeline consists of two steps: TF-IDF vectorization ('tfidf') and logistic regression classification ('clf'). This pipeline allows for the seamless application of both preprocessing (vectorization) and classification.

- Model Training: It trains the pipeline on the training data (X_train, Y_train) using the fit method.

- Predictions: It makes predictions on the test data (X_test) using the trained pipeline and stores the predictions in Y_pred.

- Model Evaluation: It evaluates the model's performance by calculating and printing the accuracy score and the classification report using accuracy_score and classification_report functions, respectively. The classification report provides metrics such as precision, recall, F1-score, and support for each class, as well as overall accuracy.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 6: LSTM Library

#### 6.1 LSTM

The code defines an LSTM (Long Short-Term Memory) neural network model for text classification using TensorFlow Keras. It starts by defining the input layer to accept string inputs. The text data is then vectorized using a pre-trained text vectorizer and embedded into dense representations. An LSTM layer with 64 units and a hyperbolic tangent activation function processes the embedded data. Finally, a dense output layer with a sigmoid activation function produces binary classification predictions. This model architecture enables capturing long-range dependencies in sequential data like text, making it suitable for tasks such as sentiment analysis or text classification.

#### 6.2 Model Compilation

The code compiles the previously defined LSTM model (model_2) using binary cross-entropy as the loss function, the Adam optimizer, and accuracy as the evaluation metric. This compilation step configures the training process by specifying how the model should learn from the provided data and how its performance should be measured during training.

#### 6.3 Model Training

This code trains the LSTM model (model_2) on the training data (X_train and Y_train) for 5 epochs while validating its performance on the test data (X_test and Y_test). The training process involves adjusting the model's weights based on the optimization algorithm and the specified loss function, with validation data used to monitor the model's performance on unseen examples during training. The history_model_2 object stores the training history, including metrics such as loss and accuracy, for later analysis or visualization.

#### 6.4 Model Prediction

The trained model (model_2) is used to make predictions on the test data (X_test). The predicted probabilities for each instance in the test data are obtained using the predict method. These probabilities are then converted into binary predictions by thresholding at 0.5, classifying instances as either positive or negative based on whether their predicted probability is greater than or equal to 0.5. Finally, a classification report is generated to evaluate the model's performance on the test data, providing metrics such as precision, recall, and F1-score for each class, as well as overall accuracy. This allows for a comprehensive assessment of the model's predictive capabilities.

#### 6.5 Classification

This code performs classification using the trained LSTM model (model_2) on the test data (X_test). It generates predictions (Y_pred) by applying the model to the test data and then converts the predicted probabilities to binary labels by thresholding at 0.5, classifying instances as positive if the predicted probability is greater than or equal to 0.5, and negative otherwise.

#### 6.6 Accuracy, Macro Average, Weighted Average

This code calculates and prints various classification metrics such as accuracy, macro average, and weighted average based on the ground truth labels (Y_test) and the predicted labels (Y_pred). The classification_report function from scikit-learn generates a detailed report containing precision, recall, F1-score, and support for each class, along with macro and weighted averages of these metrics across all classes.

#### 6.7 Plot training history

This code plots the training history of an LSTM model by displaying the changes in accuracy and loss over epochs. The first plot shows the training and validation accuracy over epochs, while the second plot shows the training and validation loss over epochs. These visualizations help in understanding the performance and convergence of the model during training.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
