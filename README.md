# Interactive-Q-A-System-Using-Llama-3.2-1B-Model-with-Hugging-Face-Integration

This repository demonstrates the creation of an **Interactive Question-Answering System** using the **Llama 3.2-1B Model** from Meta's Llama stack and its integration with **Hugging Face Transformers**. This project enables users to interact with a lightweight and efficient language model for Q&A tasks.

---

## Features

- **Llama 3.2-1B Model**: Utilizes a high-performance lightweight model for causal language modeling.
- **Hugging Face Integration**: Employs Hugging Face's `transformers` library for seamless model loading and text generation.
- **Interactive Mode**: Allows users to interact with the model through a simple console-based Q&A interface.
- **Custom Configuration**: Saves and loads pre-trained model configurations locally or on Google Drive for efficient reuse.

---

## Installation

### Prerequisites
- Python 3.10 or later
- Hugging Face Transformers
- LangChain
- Llama Stack

### Install Required Libraries
```bash
!pip install llama-stack transformers langchain
Setup
Model List
View the available models with the command:

bash
Copy code
!llama model list
Download the Model
Use the following command to download the desired model. Provide the signed URL obtained from the Llama download portal.

bash
Copy code
!llama model download --source meta --model-id Llama3.2-1B
Configure the Model
Save a config.json file to define the model's architecture:

python
Copy code
import json

config_data = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "max_position_embeddings": 131072,
    "vocab_size": 128256,
    "torch_dtype": "bfloat16"
}

with open("/root/.llama/checkpoints/Llama3.2-1B/config.json", "w") as f:
    json.dump(config_data, f, indent=4)
Usage
Interactive Q&A System
Run the following script to start an interactive console-based Q&A system:

python
Copy code
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_path = "/root/.llama/checkpoints/Llama3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set pad_token for consistent behavior
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Welcome to the Interactive Q&A System! Type 'exit' to quit.")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("Exiting... Goodbye!")
        break
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model: {answer}")
Save Model to Google Drive
Save the model for future use:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')

drive_path = '/content/drive/MyDrive/Llama3.2-1B/'
tokenizer.save_pretrained(drive_path)
model.save_pretrained(drive_path)
Example Interaction
vbnet
Copy code
Welcome to the Interactive Q&A System! Type 'exit' to quit.

You: What is the capital of India?
Model: The capital of India is New Delhi.

You: How are you?
Model: I'm an AI language model, here to assist you with your questions.

You: Exit
Exiting... Goodbye!
