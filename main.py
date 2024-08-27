import pandas as pd
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Initialize TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# Step 1: Load and Preprocess Data
file_path = 'shortjokes.csv'
data = pd.read_csv(file_path)
jokes = data['Joke'].tolist()

# Tokenize the data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
tokenized_jokes = [tokenizer.encode(joke, add_special_tokens=True) for joke in jokes]

# Step 2: Prepare Data for TensorFlow
block_size = 128
def pad_jokes(jokes, block_size):
    return [joke[:block_size] + [tokenizer.pad_token_id] * (block_size - len(joke)) for joke in jokes]

padded_jokes = pad_jokes(tokenized_jokes, block_size)
inputs = tf.ragged.constant(padded_jokes, dtype=tf.int32).to_tensor()
labels = tf.ragged.constant([joke[1:] + [tokenizer.pad_token_id] for joke in padded_jokes], dtype=tf.int32).to_tensor()

# Modify the dataset creation for TPU
GLOBAL_BATCH_SIZE = 32 * 8  # 32 per TPU core
dataset = tf.data.Dataset.from_tensor_slices({
    "input_ids": inputs,
    "attention_mask": tf.ones_like(inputs),
    "labels": labels
}).shuffle(buffer_size=10000).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)

# Wrap model creation and compilation in TPU strategy scope
with strategy.scope():
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")
    optimizer = Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=None)

# Update training loop for TPU
num_epochs = 3
steps_per_epoch = len(dataset)

# Define the train step function
@tf.function
def train_step(batch):
    def step_fn(batch):
        with tf.GradientTape() as tape:
            outputs = model(batch, training=True)
            loss = outputs.loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    per_replica_losses = strategy.run(step_fn, args=(batch,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    progbar = tf.keras.utils.Progbar(steps_per_epoch)
    for step, batch in enumerate(dataset):
        strategy.run(train_step, args=(batch,))
        progbar.update(step + 1)
    print(f"Epoch {epoch+1} completed")

print("Training completed.")

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "Why did the chicken cross the road?"
generated_text = generate_text(prompt)
print(generated_text)

# Step 5: Save the Fine-Tuned Model
model.save_pretrained('path_to_save_model')
tokenizer.save_pretrained('path_to_save_model')

