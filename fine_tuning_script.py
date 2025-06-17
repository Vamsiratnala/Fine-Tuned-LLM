# main.py

# (Optional) Install Conda in Colab - Commented out for .py script usage
# !pip install -q condacolab
# import condacolab
# condacolab.install()

# (Optional) Create a Conda environment - Handled outside the script in most cases

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load and inspect dataset
df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])
print(df.head())
print(df.shape)
print(df['label'].value_counts())
print(df['label'].unique())

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print(df['label'].value_counts())

# 2. Split data
all_labels = df['label'].tolist()
all_texts = df['message'].tolist()
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    all_texts, all_labels, test_size=0.3, stratify=all_labels, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

# 3. Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

def convert_to_tf_dataset(encodings, labels):
    return tf.data.Dataset.from_tensor_slices(
        ({'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']}, labels)
    )

train_dataset = convert_to_tf_dataset(train_encodings, train_labels)
val_dataset = convert_to_tf_dataset(val_encodings, val_labels)
test_dataset = convert_to_tf_dataset(test_encodings, test_labels)

BATCH_SIZE = 8
train_dataset = train_dataset.shuffle(len(train_labels)).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# 4. Model
model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)

# 5. Class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(class_weights_dict)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE
)

# 6. Training loop (1 epoch as in the notebook)
epochs = 1
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    batch_count = 0

    for batch in train_dataset:
        inputs, labels = batch
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            logits = outputs.logits
            per_example_loss = loss_fn(labels, logits)
            weights = tf.gather([class_weights_dict[0], class_weights_dict[1]], labels)
            weights = tf.cast(weights, dtype=tf.float32)
            weighted_loss = tf.reduce_mean(per_example_loss * weights)
        gradients = tape.gradient(weighted_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss += weighted_loss.numpy()
        batch_count += 1

    print(f"✅ Epoch {epoch+1} completed | Average Loss: {epoch_loss / batch_count:.4f}")

# 7. Validation
all_preds = []
all_labels_eval = []

for batch in val_dataset:
    inputs, labels = batch
    outputs = model(inputs, training=False)
    logits = outputs.logits
    preds = tf.argmax(logits, axis=1)
    all_preds.extend(preds.numpy())
    all_labels_eval.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels_eval = np.array(all_labels_eval)
print("📊 Classification Report:")
print(classification_report(all_labels_eval, all_preds, target_names=["ham", "spam"]))
print("🧾 Confusion Matrix:")
print(confusion_matrix(all_labels_eval, all_preds))

# 8. Test Evaluation
test_preds = []
test_labels_eval = []
for batch in test_dataset:
    inputs, labels = batch
    outputs = model(inputs, training=False)
    logits = outputs.logits
    preds = tf.argmax(logits, axis=1)
    test_preds.extend(preds.numpy())
    if isinstance(labels, tf.Tensor) and len(labels.shape) == 0:
        test_labels_eval.append(labels.numpy())
    else:
        test_labels_eval.extend(labels.numpy())
print("Test predictions:", test_preds)

# 9. Save model and tokenizer
model.save_pretrained("distilbert-sms-spam")
tokenizer.save_pretrained("distilbert-sms-spam")
