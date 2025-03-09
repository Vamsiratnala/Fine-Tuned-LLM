import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
X = list(df['message'])
y = list(pd.get_dummies(df['label'], drop_first=True)['spam'])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Tokenize data
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

# Convert to TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(8)

# Load model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train model
model.fit(train_dataset, epochs=3)

# Evaluate model
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get model predictions
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions.logits, axis=1)

# Print classification report
true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
print(classification_report(true_labels, predicted_labels))

# Save model
model.save("distilbert_model")

# Function to predict text
def predict_text(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    logits = model.predict(dict(inputs)).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    predicted_class = tf.argmax(probs, axis=-1).numpy()[0]
    class_labels = {0: "ham", 1: "spam"}
    predicted_label = class_labels[predicted_class]
    return predicted_label, probs

# Example usage
text = "This is an amazing product!"
predicted_class, probabilities = predict_text(text)
print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {probabilities}")

text = "Congratulations! You've won a $1000 gift card! 🎉 Claim your prize NOW by clicking the link below."
predicted_class, probabilities = predict_text(text)
print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {probabilities}")
