from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
max_seq_length = 2048 # Supports RoPE Scaling internally


### Model Properties
data_file = "Datasets/formatted_reflection_v2.jsonl"
new_model_name = "shiny-phi3.5"
dataset_url = data_file #Can also be huggingface address for online dataset

dataset = load_dataset("json", data_files = {"train" : dataset_url}, split = "train")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

#Model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        # per_device_train_batch_size = 2,
        # gradient_accumulation_steps = 4,
        # warmup_steps = 10,
        # max_steps = 60,
        # fp16 = not is_bfloat16_supported(),
        # bf16 = is_bfloat16_supported(),
        # logging_steps = 1,
        # output_dir = "outputs",
        # optim = "adamw_8bit",
        # seed = 3407,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 60,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer.train()

# Export model to GGUF for llama.cpp
model.save_pretrained_gguf(new_model_name, tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf(new_model_name, tokenizer, quantization_method = "q4_0")
model.save_pretrained_gguf(new_model_name, tokenizer, quantization_method = "q4_K_M")

