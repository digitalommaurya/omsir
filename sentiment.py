import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# 1. Load CSV Data
df = pd.read_csv("sentiment.csv")  # Make sure the file is in the same directory
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

# 2. Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 3. Tokenize Text
vocab_size = 1000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 4. Pad Sequences
max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# 🔧 Convert labels to numpy arrays to avoid ValueError
y_train = np.array(y_train)
y_test = np.array(y_test)

# 5. Build the Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Train the Model
model.fit(X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test))

# Save the model with proper extension
model.save("sentiment_model.keras")

# Save tokenizer
token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(token_json)

print("Model and tokenizer saved.")

# 7. Evaluate
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# 8. Load Model (use correct filename)
loaded_model = tf.keras.models.load_model("sentiment_model.keras")

# 9. Load Tokenizer
with open('tokenizer.json') as f:
    token_data = f.read()
loaded_tokenizer = tokenizer_from_json(token_data)

# 10. Predict on new samples
texts_to_predict = [
    "I really enjoyed this movie!",
    "It was a waste of time."
]

# Tokenize and pad
sequences = loaded_tokenizer.texts_to_sequences(texts_to_predict)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# Predict
predictions = loaded_model.predict(padded)

for text, pred in zip(texts_to_predict, predictions):
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"Text: {text}\nSentiment: {sentiment} (Confidence: {pred[0]:.2f})\n")
