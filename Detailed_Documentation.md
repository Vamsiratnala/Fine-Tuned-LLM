Sure! Here is the detailed documentation for your project:

---

# Fine-Tuned Sentiment Analysis with DistilBERT

This project focuses on fine-tuning a pre-trained DistilBERT model using the Hugging Face Transformers library for a custom sentiment analysis task. Specifically, it involves training the model to classify SMS messages as "ham" (non-spam) or "spam". The dataset used for this task is the SMS Spam Collection Dataset.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Tokenization](#tokenization)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the code, you need to install the necessary dependencies. Use the following commands to set up your environment:

```bash
pip install pandas tensorflow transformers scikit-learn
```

## Dataset

The dataset used in this project is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). It consists of a set of SMS messages labeled as "ham" (non-spam) and "spam".

## Preprocessing

1. **Load the Dataset**:
    - The dataset is loaded into a pandas DataFrame.
    - The dataset is expected to be in a tab-separated format with two columns: `label` and `message`.

2. **Explore the Dataset**:
    - Check the shape of the dataset to understand the number of samples.
    - Display the first few rows of the dataset to get a glimpse of the data.

3. **Split the Data**:
    - Split the dataset into independent features (`X`) and dependent labels (`y`).
    - Convert the labels to binary format where "ham" is 0 and "spam" is 1.
    - Split the data into training and testing sets using an 80-20 split.

## Tokenization

1. **Initialize the Tokenizer**:
    - Use the DistilBERT tokenizer from the Hugging Face Transformers library to tokenize the text data.

2. **Tokenize the Data**:
    - Tokenize both the training and testing sets with appropriate truncation and padding.

3. **Convert to TensorFlow Dataset**:
    - Convert the tokenized data into TensorFlow Dataset objects for efficient data handling during training.

## Training

1. **Initialize the Model**:
    - Use the DistilBERT model for sequence classification from the Hugging Face Transformers library.
    - Set up a distributed strategy for training on multiple GPUs if available.

2. **Compile the Model**:
    - Compile the model with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric.

3. **Train the Model**:
    - Train the model on the training dataset for a specified number of epochs (e.g., 3 epochs).

## Evaluation

1. **Evaluate the Model**:
    - Evaluate the trained model on the test dataset to determine its performance.
    - Calculate and display the test loss and accuracy.

2. **Get Predictions**:
    - Get the model's predictions on the test dataset.
    - Convert the logits to class labels.

3. **Classification Report**:
    - Generate a classification report to evaluate the precision, recall, and F1-score of the model.

## Saving the Model

1. **Save the Model**:
    - Save the trained model to disk for future use.

## Prediction

1. **Define Prediction Function**:
    - Define a function to make predictions on new text data.
    - The function tokenizes the input text, gets model predictions, converts logits to probabilities, and maps the predicted class index to the corresponding label.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This documentation provides a comprehensive guide to the steps involved in fine-tuning a DistilBERT model for sentiment analysis, including dataset preparation, tokenization, model training, evaluation, and making predictions.
