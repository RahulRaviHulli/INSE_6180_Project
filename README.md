# Sentiment Analysis on Reddit Comments and Posts for OSINT Purposes
Using sentiment analysis on Reddit comments and posts can provide valuable insights into detecting signs of depression within the community, potentially aiding in early intervention and support. By analyzing the sentiment expressed in these interactions, patterns indicative of depression can be identified, enabling targeted outreach and mental health resources allocation. Leveraging machine learning algorithms, sentiment analysis offers a scalable approach to monitor the emotional well-being of Reddit users and facilitate timely interventions when needed.

## Video Presentation Link: https://youtu.be/yemdYJgZUNE

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

### 7: GRU Library

#### 7.1 GRU Model Definition

The code defines a GRU (Gated Recurrent Unit) neural network model for text classification. It starts with an input layer accepting string inputs. The inputs are then vectorized using a text vectorizer and embedded into dense representations. A GRU layer with 64 units and hyperbolic tangent activation function processes the embedded inputs. Finally, a Dense layer with sigmoid activation produces binary classification outputs.

#### 7.2 Model Compilation

The code compiles the GRU model using binary cross-entropy loss and the Adam optimizer. Binary cross-entropy is suitable for binary classification tasks, while the Adam optimizer efficiently updates the model's weights during training. Accuracy is chosen as the metric to evaluate the model's performance.

#### 7.3 Model Training:

This section involves training the GRU model on the training dataset while validating its performance on the test dataset over 5 epochs. The training history is stored for further analysis. This code snippet trains the GRU model using the training data (X_train and Y_train) and validates it on the test data (X_test and Y_test) for 5 epochs. The training history is stored in the variable history_gru.

#### 7.4 Model Accuracy, Macro Average, Weighted Average:

In this step, the trained GRU model is utilized to make predictions on the test dataset. The model predicts the probabilities of the samples belonging to the positive class (1). These probabilities are then converted into binary predictions by considering a threshold of 0.5. Finally, the classification report is printed to assess the performance of the model based on its predictions compared to the ground truth labels.

#### 7.5 Plot Training:

This section presents visualizations of the training progress of the GRU model. Two plots are generated: one illustrating the model accuracy on both the training and validation datasets across different epochs, and the other showing the corresponding loss values. These plots help in understanding how the model's performance evolves during the training process and whether it is overfitting or underfitting.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 8: Custom LSTM Function

#### 8.1 Define Custom LSTM

The CustomLSTM class defines a custom implementation of the Long Short-Term Memory (LSTM) layer in TensorFlow/Keras. Here's what's happening in the code:

- Initialization: The constructor (__init__ method) initializes the LSTM layer with the specified number of units and whether to return sequences for each time step.

- Building the Layer: The build method initializes the weights and biases for the LSTM gates (Forget Gate, Input Gate, Candidate Cell State, and Output Gate). These weights and biases are trainable parameters learned during training.

- Forward Pass: The call method performs the forward pass of the LSTM layer. It iterates over each time step in the input sequence, computes the LSTM gates and cell state updates, and computes the hidden state output.

- LSTM Gates: The LSTM gates (Forget Gate, Input Gate, and Output Gate) control the flow of information through the cell state. They are computed using the input at the current time step (x_t), the previous hidden state (hidden_states), and the learned weights (Wf, Wi, Wo) and biases (bf, bi, bo).

- Cell State Update: The cell state (cell_states) is updated based on the forget gate (f_t), input gate (i_t), and candidate cell state (c_tilda_t). This helps the LSTM layer to remember or forget information over time.

- Hidden State Calculation: The hidden state (hidden_states) is computed using the output gate (o_t) and the updated cell state. This is the output of the LSTM layer.

- Return Sequences: If return_sequences is set to True, the method returns the hidden states for each time step. Otherwise, it returns only the hidden state for the last time step.

This custom LSTM layer provides a flexible and customizable implementation of the LSTM architecture, allowing for experimentation with different configurations and adaptations to specific use cases.

#### 8.2 Model Definition and Compilation

Model Definition:

We define the input layer with the shape of (max_length,) using tf.keras.layers.Input. The input data is passed through the embedding layer (embedding) to convert numerical indices into dense vectors. The embedded sequences are then passed through the custom LSTM layer (CustomLSTM) with 64 units and return_sequences=False, indicating that only the final hidden state is returned. Finally, a dense layer with a sigmoid activation function is added to produce the output predictions. The model is instantiated using tf.keras.Model, with the input and output layers specified. Model Compilation:

We compile the model using model.compile, specifying 'binary_crossentropy' as the loss function for binary classification. The Adam optimizer is used for optimization, and we include 'accuracy' as a metric to monitor during training.

#### 8.3 Preprocess and tokenize the input data

Tokenization: We tokenize the training and test data using the text_vectorizer function. This function converts text data into numerical sequences suitable for input into the model. Model Training:

We train the custom LSTM model (model_custom_lstm) using the tokenized training data (X_train_tokenized) and corresponding labels (Y_train). During training, we validate the model's performance on the tokenized test data (X_test_tokenized) and corresponding labels (Y_test). Training is conducted over 5 epochs, during which the model learns to map tokenized sequences to their corresponding labels.

#### 8.4 Live Prediction

Function Definition (predict_depression):

The function takes two arguments: model (the trained model) and text_vectorizer (a function for vectorizing input text). Inside the function, the user is prompted to enter a text input. The entered text is then passed through a cleaning function (clean) to preprocess it, assuming such a function has been defined elsewhere. The cleaned text is then converted into a list and passed through the text_vectorizer function to convert it into a numerical vector suitable for input to the model. The model then predicts the class probability of the input text being associated with depression. If the predicted probability is greater than or equal to 0.5, the function prints "The input text represents depression." Otherwise, it prints "The input text does not represent depression." Live Prediction:

The predict_depression function is called with the trained model_custom_lstm and text_vectorizer function as arguments to perform live predictions based on user input. Generating Predictions:

After performing live predictions, the same model (model_custom_lstm) is used to generate predictions on the test data (X_test_tokenized). These predictions are then converted into binary labels based on a threshold of 0.5, where any prediction greater than or equal to 0.5 is classified as 1 (representing depression), and anything below 0.5 is classified as 0 (not representing depression).

#### 8.5 Generate and Print Classification Report

The classification_report function is called with two arguments: Y_test (the actual labels of the test dataset) and y_pred_labels (the predicted labels generated by the model). This function computes various metrics such as precision, recall, F1-score, and support for each class (in this case, binary classes: depression and non-depression). The classification report summarizes these metrics for each class and also provides macro and weighted averages across all classes. Print Classification Report:

The generated classification report is stored in the variable report. The classification report is printed to the console using the print function, preceded by a header "Classification Report:". The printed report provides insights into the performance of the model across different metrics and classes, aiding in the evaluation of its performance.

#### 8.6 Plot Training

Plotting Model Accuracy:

The plt.plot() function is used to plot two lines: one for training accuracy (history.history['accuracy']) and another for validation accuracy (history.history['val_accuracy']). The label parameter is set for each line to differentiate between training and validation accuracy. Title, xlabel, ylabel, and legend are added to the plot for clarity. Finally, plt.show() is called to display the plot. Plotting Model Loss:

Similar to accuracy plotting, the plt.plot() function is used to plot training loss (history.history['loss']) and validation loss (history.history['val_loss']). Labels, title, xlabel, ylabel, and legend are added as before. The plot is displayed using plt.show(). These plots are essential for visualizing the training progress of the custom LSTM model. They help in understanding how the model's accuracy and loss evolve over epochs, providing insights into its performance and potential areas for improvement.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 9: Custom GRU Function

#### 9.1 Define Custom GRU Layer

Initialization:

The init method initializes the custom GRU layer with the specified number of units and whether it should return sequences for each time step. Build Method:

In the build method, weights for the update gate (z), reset gate (r), and candidate hidden state are initialized. Weights are created using add_weight with appropriate shapes and initializers. Call Method:

The call method defines the forward pass of the custom GRU layer. For each time step, the input is processed through the update gate (z), reset gate (r), and candidate hidden state (h_tilda). The hidden state is updated based on the update gate and the candidate hidden state using the GRU update equations. The updated hidden states are stored in a list for each time step. Finally, if return_sequences is True, the list of hidden states is stacked along the time step axis and returned; otherwise, only the last hidden state is returned. This custom GRU layer allows for flexibility in building GRU-based neural network architectures and can be used as a drop-in replacement for the built-in GRU layer provided by TensorFlow/Keras.

#### 9.2 Model Definition and Compilation for GPU

Model Definition for GRU:

An input layer is defined with the shape (max_length,), where max_length represents the length of the input sequences. The input is passed through an embedding layer to convert the input sequences into dense vectors. The output of the embedding layer is fed into a custom GRU layer with 64 units. The return_sequences parameter is set to False, indicating that only the final hidden state of the GRU will be returned. Finally, a Dense layer with a sigmoid activation function is added to produce the model's output.

Model Compilation for GRU:

The model is compiled with binary cross-entropy loss and the Adam optimizer. Metrics for evaluation are set to accuracy, which measures the proportion of correctly classified samples. Once the model is defined and compiled, it is ready for training on labeled data for the task of binary classification.

#### 9.3 Preprocess and tokenize the input data for GRU

In this step, the input data is preprocessed and tokenized specifically for the custom GRU model. Here's what's happening:

Preprocessing and Tokenization:

The training and test data are tokenized using the text_vectorizer function. This function converts raw text data into numerical sequences suitable for input to the model.
The tokenization process involves converting each word in the text data into a unique numerical token. This enables the model to process the textual information effectively.
Model Training for GRU:

The tokenized training data (X_train_tokenized_gru) and corresponding labels (Y_train) are used to train the custom GRU model.
The validation data (X_test_tokenized_gru) and labels (Y_test) are provided to evaluate the model's performance during training.
The model is trained for 5 epochs, allowing it to learn patterns and relationships within the data.
After training, the model's training history (history_custom_gru) is stored for further analysis and visualization of performance metrics such as accuracy and loss over epochs.

#### 9.4 Live Prediction for GRU

Function Definition:

The function predict_depression_gru takes two arguments: the trained GRU model (model_custom_gru) and the text vectorizer function (text_vectorizer). Inside the function, the user is prompted to enter a text input. The entered text is cleaned using a clean function (assuming it's defined elsewhere) to preprocess the input. The cleaned text is tokenized using the text_vectorizer function to convert it into a numerical sequence. The model predicts the depression likelihood based on the input text sequence. If the predicted probability is greater than or equal to 0.5, it's classified as representing depression; otherwise, it's classified as not representing depression. Live Prediction:

The predict_depression_gru function is called with the trained GRU model (model_custom_gru) and the text vectorizer function (text_vectorizer) as arguments. The user is prompted to enter a text, and the model predicts whether the input text represents depression or not. Generate Predictions:

Predictions are generated for the test data (X_test_tokenized_gru) using the trained GRU model (model_custom_gru). Predicted probabilities are thresholded at 0.5 to obtain binary predictions (y_pred_labels_gru).

#### 9.5 Generate and Print Classification report for GRU

Generate Classification Report:

The classification_report function from Scikit-learn is used to generate a classification report based on the true labels (Y_test) and the predicted labels (y_pred_labels_gru) obtained from the GRU model. Print Classification Report:

The classification report for the GRU model is printed to the console. The report provides metrics such as precision, recall, F1-score, and support for each class, along with averages. This classification report provides detailed insights into the performance of the GRU model in classifying the test data, including metrics for both positive and negative classes, aiding in understanding the model's strengths and weaknesses.

#### Plotting Model Accuracy:

The training and validation accuracies over epochs are plotted. The accuracy and val_accuracy values from the history_custom_gru object are used. The plot provides insights into how the accuracy of the model changes over training epochs.

Plotting Model Loss:

The training and validation losses over epochs are plotted. The loss and val_loss values from the history_custom_gru object are used. The plot helps in understanding how the loss of the model evolves during training, indicating whether the model is learning effectively. These plots offer valuable information about the training dynamics of the GRU model, enabling assessment of its performance and convergence behavior.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###  Plotting for Comparisons

#### 10: Plotting for Comparisons

Plotting Model Accuracy:

Here, we visualize the training and validation accuracy of three different models: LSTM, GRU, Custom LSTM and Custom GRU. The training accuracy represents how well the models perform on the training data during the training process, while the validation accuracy indicates their performance on unseen validation data. The purpose of plotting these accuracies is to assess how well each model is learning from the data and whether they are overfitting or underfitting.

Training Accuracy:

LSTM Training Accuracy: Accuracy of the LSTM model on the training data.
GRU Training Accuracy: Accuracy of the GRU model on the training data.
Custom LSTM Training Accuracy: Accuracy of the Custom LSTM model on the training data.
Custom GRU Training Accuracy: Accuracy of the Custom GRU model on the training data.
Validation Accuracy:

LSTM Validation Accuracy: Accuracy of the LSTM model on the validation data.
GRU Validation Accuracy: Accuracy of the GRU model on the validation data.
Custom LSTM Validation Accuracy: Accuracy of the Custom LSTM model on the validation data.
Custom GRU Validation Accuracy: Accuracy of the Custom GRU model on the validation data.
Plotting Model Loss:

Similar to accuracy, this section visualizes the training and validation loss of the three models: LSTM, GRU, Custom LSTM and Custom GRU. Loss represents the error between the actual and predicted values. Lower loss values indicate better performance. By plotting these losses, we can monitor how well each model is learning and whether they are overfitting or underfitting.

Training Loss:

LSTM Training Loss: Loss of the LSTM model on the training data.
GRU Training Loss: Loss of the GRU model on the training data.
Custom LSTM Training Loss: Loss of the Custom LSTM model on the training data.
Custom GRU Training Loss: Loss of the Custom GRU model on the training data.
Validation Loss:

LSTM Validation Loss: Loss of the LSTM model on the validation data.
GRU Validation Loss: Loss of the GRU model on the validation data.
Custom LSTM Validation Loss: Loss of the Custom LSTM model on the validation data.
Custom GRU Validation Loss: Loss of the Custom GRU model on the validation data.
###
