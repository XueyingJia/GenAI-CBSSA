from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b")

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, "XueyingJia/pythia-1b-sft-SG")

# Merge LoRA weights into the base model
model = model.merge_and_unload()

# Save the merged model to a directory
tokenizer = AutoTokenizer.from_pretrained("XueyingJia/pythia-1b-sft-SG")

# Save the combined model
model.push_to_hub("XueyingJia/pythia-1b-sft-SG-merge")
tokenizer.push_to_hub("XueyingJia/pythia-1b-sft-SG-merge")