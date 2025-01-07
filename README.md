# Interactive-Q-A-System-Using-Llama-3.2-1B-Model-with-Hugging-Face-Integration

# 🌟 Interactive Q&A System with Llama 3.2-1B 🌟

Welcome to the **Interactive Question-Answering System** powered by **Meta's Llama 3.2-1B** model and **Hugging Face Transformers**! 🚀 This project is a hands-on showcase of how cutting-edge language models can be seamlessly integrated for building interactive and efficient Q&A systems. 

---

## 🛠️ Features at a Glance

✨ **State-of-the-Art Model**: Uses Llama 3.2-1B for efficient language understanding and generation.  
✨ **Interactive Console**: Ask questions in real time and get dynamic responses!  
✨ **Customizable**: Tailor the system to meet specific requirements (e.g., larger context, model fine-tuning).  
✨ **Efficient Storage**: Save and reuse your model locally or in Google Drive.  

---

## ⚡ Quick Start

### 🔧 Prerequisites

Ensure you have the following:
- **Python 3.10+**
- Required Python libraries: `llama-stack`, `transformers`, and `langchain`

Install them with:
```bash
pip install llama-stack transformers langchain
🚀 Getting Started
Step 1️⃣: List Available Models
Discover available Llama models:

bash
Copy code
!llama model list
Step 2️⃣: Download the Model
Get the Llama 3.2-1B model with your unique signed URL:

bash
Copy code
!llama model download --source meta --model-id Llama3.2-1B
Step 3️⃣: Configure the Model
Set up the configuration for optimal performance:

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

print("Configuration saved!")
💬 Interactive Q&A
Run the following script to start your interactive Q&A session:

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

print("🤖 Welcome to the Interactive Q&A System! Type 'exit' to quit.")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("👋 Exiting... Goodbye!")
        break
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"💡 Model: {answer}")
🧳 Save the Model to Google Drive
Want to reuse your model later? Save it to Google Drive:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')

drive_path = '/content/drive/MyDrive/Llama3.2-1B/'
tokenizer.save_pretrained(drive_path)
model.save_pretrained(drive_path)

print("🎉 Model saved to Google Drive!")
📚 Example Interaction
vbnet
Copy code
🤖 Welcome to the Interactive Q&A System! Type 'exit' to quit.

You: What is the capital of India?  
💡 Model: The capital of India is New Delhi.  

You: Who is the president of the United States?  
💡 Model: The current president of the United States is Joe Biden (as of 2024).  

You: Exit  
👋 Exiting... Goodbye!  
🎯 Customization Options
🌈 Tailor the experience to your needs:

Fine-tuning: Adapt the model to your domain-specific tasks.
Model Variants: Experiment with larger or instruction-tuned variants like Llama3.2-3B-Instruct.
Context Length: Adjust the input length for handling longer questions or conversations.
🔗 Helpful Resources
📖 Meta's Llama Documentation
📦 Hugging Face Transformers
🛠️ LangChain for Pipelines
