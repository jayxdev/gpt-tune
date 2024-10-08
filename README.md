# GPT-Tune: Fine-tuning GPT-2 for Joke Generation

This project fine-tunes a GPT-2 medium model on a dataset of jokes to create a custom joke generator. It utilizes TensorFlow and the Hugging Face Transformers library, optimized for TPU training.

## Features

- Data preprocessing and tokenization using GPT-2 tokenizer
- TPU-optimized training setup
- Fine-tuning GPT-2 medium model on a jokes dataset
- Text generation functionality with customizable parameters

## Requirements

- TensorFlow
- Hugging Face Transformers
- pandas
- TPU environment (Google Colab or Cloud TPU)

## Usage

1. Prepare your joke dataset in CSV format with a 'Joke' column.
2. Update the `file_path` variable with your dataset location.
3. Run the script to fine-tune the model.
4. Use the `generate_text()` function to generate new jokes.

## Model Training

The script performs the following steps:
1. Load and preprocess the joke data
2. Tokenize the jokes using GPT-2 tokenizer
3. Prepare the data for TensorFlow and TPU
4. Initialize and compile the GPT-2 model
5. Train the model for a specified number of epochs

## Text Generation

After training, you can generate new jokes using the `generate_text()` function:


prompt = "Why did the chicken cross the road?"
generated_text = generate_text(prompt)
print(generated_text)


## Saving the Model
The fine-tuned model and tokenizer are saved to the specified path:

model.save_pretrained('path_to_save_model')
tokenizer.save_pretrained('path_to_save_model')

## Customization
You can adjust various parameters such as:

- Model size (currently set to "gpt2-medium")
- Training hyperparameters (learning rate, batch size, etc.)
- Text generation parameters (temperature, top_k, top_p, etc.)