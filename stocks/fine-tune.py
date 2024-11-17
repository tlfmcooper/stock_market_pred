import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np
from typing import Optional, List

import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# Load and prepare the model
def prepare_model(model_name="google/gemma-2-2b"):
    # Load model with quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Binary classification
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    return model


# Prepare dataset
def prepare_dataset(texts, labels):
    return Dataset.from_dict({"text": texts, "label": labels})


# Tokenization function
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


class CustomTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss)

        return loss.detach()

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        model = self.model
        model.eval()

        # Initialize metrics, etc.
        self.callback_handler.eval_dataloader = dataloader

        observed_num_examples = 0
        all_losses = []

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            all_losses.append(loss.detach())

            if inputs.get("labels") is not None:
                observed_num_examples += len(inputs["labels"])

        metrics = {
            f"{metric_key_prefix}_loss": torch.mean(torch.stack(all_losses)).item(),
        }

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=observed_num_examples,
        )


# Modified training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=False,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
# Load and preprocess the dataset
import json


def load_dataset(filename):
    texts, labels = [], []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["Instruction"])
            labels.append(data["Response"])
    return texts, np.array(labels)


# Main training function
def train_model(train_texts, train_labels, val_texts, val_labels):
    # Prepare datasets
    train_dataset = prepare_dataset(train_texts, train_labels)
    val_dataset = prepare_dataset(val_texts, val_labels)

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Prepare model
    model = prepare_model()

    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    return trainer, model


# Load your data
train_texts, train_labels = load_dataset("./train_dataset.jsonl")
val_texts, val_labels = load_dataset("./test_dataset.jsonl")

# Convert labels to numpy arrays if they aren't already
train_labels = np.array(train_labels, dtype=np.float32)
val_labels = np.array(val_labels, dtype=np.float32)

# Train the model
trainer, model = train_model(train_texts, train_labels, val_texts, val_labels)

# Save the fine-tuned model to Hugging Face Hub
model_name = (
    "tlfmcooper/gemma-2-2b-ft-market-news"  # Replace with your desired model name
)
model.push_to_hub(model_name, use_auth_token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.push_to_hub(model_name, use_auth_token=hf_token)


def load_fine_tuned_model(model_name, use_4bit=True, num_labels=1):
    """
    Load the fine-tuned model from Hugging Face Hub
    Args:
        model_name: Name of the model on HuggingFace Hub
        use_4bit: Whether to load in 4-bit quantization
        num_labels: Number of labels for classification
    Returns:
        model, tokenizer
    """
    if use_4bit:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=hf_token,
            num_labels=num_labels,  # Specify the number of labels
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            use_auth_token=hf_token,
            num_labels=num_labels,  # Specify the number of labels
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    return model, tokenizer


def predict(text, model=None, threshold=0.5, tokenizer=None, model_name=None):
    """
    Make predictions using the fine-tuned model and return binary output
    Args:
        text: Input text to classify
        model: Fine-tuned model (optional if model_name is provided)
        threshold: Classification threshold (default: 0.5)
        tokenizer: Optional tokenizer (will be loaded if not provided)
        model_name: Name of the model on HuggingFace Hub (optional if model is provided)
    Returns:
        tuple: (binary prediction, probability)
    """
    if model is None and model_name is not None:
        model, tokenizer = load_fine_tuned_model(model_name)
    elif tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    # Prepare input
    inputs = tokenizer(
        text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    ).to(model.device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply sigmoid to get probability between 0 and 1
        probability = torch.sigmoid(outputs.logits.squeeze()).item()
        # Convert to binary prediction
        prediction = 1 if probability >= threshold else 0

    return prediction, probability  # Return both binary prediction and raw probability


# Example inference
test_text = "Donald trump election has unleashed a bull market"
model_name = "tlfmcooper/gemma-2-2b-ft-market-news"
binary_pred, prob = predict(test_text, model_name=model_name)
print(f"Binary prediction: {binary_pred}, Probability: {prob:.3f}")
