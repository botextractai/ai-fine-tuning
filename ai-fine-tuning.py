import os
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

# Hugging Face API key login
login(token = "REPLACE_THIS_WITH_YOUR_FREE_HUGGING_FACE_WRITE_API_KEY")

# Weights & Biases (wandb.ai) API key
os.environ["WANDB_API_KEY"] = "REPLACE_THIS_WITH_YOUR_FREE_WAND_API_KEY"

# Start Weights & Biases (wandb.ai)
wandb.init(project="finetuning")

# Terminate the previous Weights & Biases (wandb.ai) training run, if it is still active
if wandb.run is not None:
  wandb.finish()

# Configure the dataset
dataset_name = "squad_v2"
dataset = load_dataset(dataset_name, split="train")
eval_dataset = load_dataset(dataset_name, split="validation")

# Load the dataset and show the training and validation parts
load_dataset(dataset_name)

dataset.features

# Configure and load the model
model_id = "openlm-research/open_llama_3b_v2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
device_map="auto"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False

base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set up a tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Output directory for model checkpoints and logs is current working directory plus "results"
output_dir = os.path.join(os.getcwd(), "results")
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

new_model_name = f"openllama-3b-peft-{dataset_name}"

# Set up the training configuraton using LoRA
    # If the "push_to_hub" parameter is set to "True" and the "output_dir" is set 
    # to "new_model_name", then the model checkpoints will regularly be saved to 
    # Hugging Face during training. The "new_model_name" will be the repository name 
    # under which the model will become available on https://huggingface.co/models . 
    # The following settings save the training model checkpoints locally.
training_args = TrainingArguments(
    output_dir=output_dir,  # use new_model_name if saving to Hugging Face
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=2000,  # very high, but training can still improve after many steps!
    num_train_epochs=100,
    evaluation_strategy="steps",
    eval_steps=5,  # update steps between two evaluations
    save_total_limit=5,  # only last 5 models are saved
    push_to_hub=False,  # you can set this to true if you want to upload your model to Hugging Face Space
    load_best_model_at_end=True,  # to use in combination with early stopping
    report_to="wandb"
)

# Training configuration
max_seq_length = 512

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="question",  # this depends on the dataset!
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=200)]
)

# Training
trainer.train()

# Save the final checkpoint for re-loading:
trainer.model.save_pretrained(
    os.path.join(output_dir, "final_checkpoint"),
)

# Manually pushing the new model adapter to Hugging Face
trainer.model.push_to_hub(
    repo_id=new_model_name
)
