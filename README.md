# Fine-Tuned Sentiment Analysis with DistilBERT

This project focuses on fine-tuning a pre-trained DistilBERT model using the Hugging Face Transformers library for a custom sentiment analysis task. Specifically, it involves training the model to classify SMS messages as "ham" (non-spam) or "spam". The dataset used for this task is the SMS Spam Collection Dataset. The repository provides all the necessary code and instructions to preprocess the data, fine-tune the model, evaluate its performance, and use it for prediction.

```markdown
# Fine-Tuned Sentiment Analysis with DistilBERT

This repository contains code for fine-tuning the DistilBERT model on a custom sentiment analysis task. The dataset used is a collection of SMS messages labeled as "ham" or "spam".

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the code, you need to install the necessary dependencies. Use the following commands to set up your environment:

```bash
pip install pandas tensorflow transformers scikit-learn
```

## Dataset

The dataset used in this project is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). It consists of a set of SMS messages labeled as "ham" (non-spam) and "spam".

## Model

We use the DistilBERT model from the Hugging Face Transformers library for this task. DistilBERT is a smaller, faster, and cheaper version of BERT that retains 97% of BERT's performance.

## Training

The training process involves the following steps:
1. Load and preprocess the dataset.
2. Tokenize the text data using the DistilBERT tokenizer.
3. Split the data into training and test sets.
4. Fine-tune the DistilBERT model on the training data.
5. Save the trained model.

## Evaluation

After training, the model is evaluated on the test set to determine its performance. The evaluation metrics include accuracy, precision, recall, and F1-score.

## Usage

To use the trained model for sentiment analysis, you can run the `predict_text` function, which takes a text input and returns the predicted label ("ham" or "spam") and the class probabilities.

Example:

```python
text = "Congratulations! You've won a $1000 gift card! 🎉 Claim your prize NOW by clicking the link below."
predicted_class, probabilities = predict_text(text)

print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {probabilities}")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This structure covers the essential aspects of your project and provides users with the information needed to understand, install, and use your code. Feel free to customize this template to better fit your specific project and requirements.
