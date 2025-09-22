import torch
import os
import json
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config():
    """
    Load configuration from environment variables.
    """
    config = {
        # Model Configuration
        "model_name": os.getenv("MODEL_NAME", "gpt2"),
        "use_gpu": os.getenv("USE_GPU", "false").lower() == "true",
        # Dataset Configuration
        "dataset_path": os.getenv("DATASET_PATH", "simple_dataset.jsonl"),
        # Training Configuration
        "output_dir": os.getenv("OUTPUT_DIR", "./sft_model"),
        "num_train_epochs": int(os.getenv("NUM_TRAIN_EPOCHS", "1")),
        "per_device_train_batch_size": int(
            os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1")
        ),
        "gradient_accumulation_steps": int(
            os.getenv("GRADIENT_ACCUMULATION_STEPS", "16")
        ),
        "learning_rate": float(os.getenv("LEARNING_RATE", "2e-4")),
        "warmup_ratio": float(os.getenv("WARMUP_RATIO", "0.05")),
        "logging_steps": int(os.getenv("LOGGING_STEPS", "10")),
        "save_steps": int(os.getenv("SAVE_STEPS", "500")),
        "eval_steps": int(os.getenv("EVAL_STEPS", "500")),
        # LoRA Configuration
        "lora_r": int(os.getenv("LORA_R", "16")),
        "lora_alpha": int(os.getenv("LORA_ALPHA", "32")),
        "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.05")),
        # Generation Configuration
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "100")),
        "do_sample": os.getenv("DO_SAMPLE", "false").lower() == "true",
    }

    return config


def get_target_modules(model_name):
    """
    Get appropriate target modules for LoRA based on model architecture.
    """
    if "gpt2" in model_name.lower():
        return ["c_attn", "c_proj"]
    elif "llama" in model_name.lower():
        return [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif "mistral" in model_name.lower():
        return [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif "qwen" in model_name.lower():
        return [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        # Default to common transformer modules
        return [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


def generate_responses(
    model, tokenizer, user_message, max_new_tokens=100, do_sample=True
):
    """
    Generate responses using the fine-tuned model.
    """
    # Format input
    formatted_input = f"Instruction: {user_message}\nOutput:"

    # Tokenize input
    inputs = tokenizer(formatted_input, return_tensors="pt")

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
        )

    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    response = response.strip()

    # Configuration for response truncation
    truncation_config = {
        "default_max_words": 50,
        "content_types": {
            "code": {
                "keywords": ["code", "function", "def ", "import", "class", "```"],
                "max_words": 80,
            },
            "short": {
                "keywords": ["yes", "no", "simple", "basic", "ok"],
                "max_words": 30,
            },
            "technical": {
                "keywords": ["algorithm", "model", "training", "neural"],
                "max_words": 60,
            },
        },
        "stop_tokens": [".", "!", "?", "\n\n", "```", "---", "##"],
        "ellipsis": "...",
    }

    words = response.split()
    max_words = truncation_config["default_max_words"]

    response_lower = response.lower()
    for content_type, config in truncation_config["content_types"].items():
        if any(keyword in response_lower for keyword in config["keywords"]):
            max_words = config["max_words"]
            break

    if len(words) > max_words:
        for i in range(max_words, len(words)):
            if words[i] in truncation_config["stop_tokens"]:
                response = " ".join(words[: i + 1])
                break
        else:
            response = " ".join(words[:max_words])
            if not response.endswith((".", "!", "?")):
                response += truncation_config["ellipsis"]

    return response


def test_model_with_questions(
    model, tokenizer, questions, title="Model Output", config=None
):
    """
    Test the model with a list of questions.
    """
    print(f"\n{title}")
    print("=" * 50)

    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        response = generate_responses(
            model,
            tokenizer,
            question,
            max_new_tokens=config.get("max_new_tokens", 100),
            do_sample=config.get("do_sample", False),
        )
        print(f"A{i}: {response}")


def load_model_and_tokenizer(model_name, use_gpu=False):
    """
    Load model and tokenizer with optional GPU support.
    """
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def display_dataset(dataset):
    """
    Display dataset information and sample entries.
    """
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")

    print("\nSample entries:")
    for i in range(min(3, len(dataset))):
        print(f"\nEntry {i+1}:")
        for key, value in dataset[i].items():
            print(f"  {key}: {value}")


def generate_prompt(data_point):
    """
    Generate a simple prompt from instruction, input, and output.
    """
    if (
        data_point["input"]
        and data_point["input"] != ""
        and data_point["input"] != "Not applicable"
    ):
        text = f"Instruction: {data_point['instruction']}\nInput: {data_point['input']}\nOutput: {data_point['output']}"
    else:
        text = (
            f"Instruction: {data_point['instruction']}\nOutput: {data_point['output']}"
        )

    return text


def format_dataset_for_sft(dataset):
    """
    Format dataset for SFT with a very simple approach.
    """
    # Generate prompts
    text_column = [generate_prompt(data_point) for data_point in dataset]

    # Create a new dataset with only the text field
    formatted_data = [{"text": text} for text in text_column]
    formatted_dataset = Dataset.from_list(formatted_data)

    print("Sample formatted text:")
    print(formatted_dataset[0]["text"])
    print("\n" + "=" * 50 + "\n")

    return formatted_dataset


def load_dataset_from_path(dataset_path):
    """
    Load dataset from file path (JSONL or Hugging Face dataset).
    """
    print(f"Loading dataset from {dataset_path}")

    if dataset_path.endswith(".jsonl"):
        # Load JSONL dataset
        data = []
        with open(dataset_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        dataset = Dataset.from_list(data)
    elif dataset_path.endswith(".json"):
        # Load JSON dataset
        with open(dataset_path, "r") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        # Load Hugging Face dataset
        dataset = load_dataset(dataset_path, split="train")

    return dataset


def run_sft_pipeline(config):
    """
    Run SFT pipeline with configuration from environment variables.
    """
    print("=== SFT Pipeline with QLoRA ===")
    print(f"Model: {config['model_name']}")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Use GPU: {config['use_gpu']}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config["model_name"], config["use_gpu"])

    # Load dataset
    dataset = load_dataset_from_path(config["dataset_path"])
    display_dataset(dataset)

    # Format dataset for SFT
    print("\nFormatting dataset for SFT...")
    dataset = format_dataset_for_sft(dataset)

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=1234)

    # Split into train and test
    dataset = dataset.train_test_split(test_size=0.2)
    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Apply LoRA configuration
    print("\nApplying LoRA configuration...")
    target_modules = get_target_modules(config["model_name"])
    print(f"Target modules for {config['model_name']}: {target_modules}")

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=False,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    # Initialize SFTTrainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=lambda example: example["text"],
    )

    # Train the model
    print("\nStarting training...")
    trainer.train()

    # Save the model
    final_output_dir = f"{config['output_dir']}_final"
    print(f"\nSaving model to {final_output_dir}")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print("\nTraining completed successfully!")

    return trainer.model, tokenizer


def main():
    """
    Main function to run the SFT pipeline.
    """
    # Load configuration
    config = load_config()

    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Test questions
    test_questions = [
        "Explain the concept of gradient descent in machine learning.",
        "What is the difference between supervised and unsupervised learning?",
        "How does a neural network work?",
    ]

    # Run the pipeline
    try:
        model, tokenizer = run_sft_pipeline(config)

        # Test the model
        test_model_with_questions(model, tokenizer, test_questions, config=config)

    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("Make sure you have a .env file with proper configuration.")
        print("You can copy env_example.txt to .env and modify it as needed.")


if __name__ == "__main__":
    main()
