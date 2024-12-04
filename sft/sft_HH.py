# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# Full training
python examples/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub

# LoRA
python examples/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub

# QLoRA SG
python sft.py \
    --model_name_or_path EleutherAI/pythia-160m \
    --attn_implementation flash_attention_2 \
    --load_in_8bit True \
    --device_map auto \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing False\
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --lora_task_type CAUSAL_LM \
    --output_dir pythia-160m-sft-HH \
    --push_to_hub
"""
from transformers.utils import is_bitsandbytes_available
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from datasets import load_dataset
import copy

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM
)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    
    print("Initial model_config:", vars(model_config))

    clean_config = {
        'load_in_8bit': model_config.load_in_8bit,
        'load_in_4bit': model_config.load_in_4bit,
        'bnb_4bit_quant_type': model_config.bnb_4bit_quant_type if model_config.load_in_4bit else None
    }

    clean_config = {k: v for k, v in clean_config.items() if v is not None}

    # Create quantization config directly
    quantization_config = BitsAndBytesConfig(**clean_config) if model_config.load_in_8bit or model_config.load_in_4bit else None
    print("Quantization config:", quantization_config)

    if quantization_config is not None:
        config_dict = {
            k: v for k, v in quantization_config.to_dict().items() 
            if not k.startswith('_') and k != 'quant_method'
        }
        training_args.quantization_config = config_dict
        print("Training args quantization config:", training_args.quantization_config)

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map="auto",
        quantization_config=quantization_config,
    )

    print("Model kwargs:", {k: str(v) for k, v in model_kwargs.items() if v is not None})

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=model_config.trust_remote_code, 
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token


    def tokenize(element):
        return tokenizer(element["chosen"], truncation=True, padding="max_length", max_length=1024)

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    dataset = dataset.map(tokenize, batched=True)
    

    instruction_template = "Human:"
    response_template = "Assistant:"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    # Load one of the harmless subsets
    

    # Modify SFTTrainer initialization to use the processed configs
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collator,
        eval_dataset=dataset["test"] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config)
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
