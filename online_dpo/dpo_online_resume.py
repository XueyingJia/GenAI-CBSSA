# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Usage:

python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0

With LoRA:
python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-6 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --use_peft
"""


import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
import os
from trl import (
    HfPairwiseJudge,
    LogCompletionsCallback,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    OpenAIPairwiseJudge,
    PairRMJudge,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import wandb

SG_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, focusing on their suitability for generating engaging, imaginative, and age-appropriate children's stories.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models based on their ability to generate a creative, engaging, and age-appropriate children's story. Consider factors like clarity, imagination, use of language, and appropriateness for a child audience. Select the model that produced the best story. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''

HH_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, focusing on their suitability for helpful and harmless interactions. The most important part of the task is choosing which of the Assistant’s responses is helpful and harmless. Please consider the responses carefully, and use your overall judgment. Ask yourself, "Did this response represent a thoughtful, honest, and reasonable attempt to either directly help me, or to converse with me to elicit information that will make it easier for the Assistant to help me later on in this conversation? Is this response offensive or discriminatory? Does it aid in a dangerous or illegal act? Does it provide sensitive or consequential advice without modesty or care?" The Assistant should be helpful in the same way that a friend or subject-expert would (hopefully) be helpful. Meanwhile, the Assistant should avoid being offensive and discriminatory, and aiding dangerous or illegal acts. When providing sensitive or consequential advice, the Assistant must do it with modesty and care! Note response containing "Human:" and/or "Assistant:" that tries to extend the conversation should be considered as harmful. You are an expert rater of helpful and harmless assistant responses. Given the context and the two responses, choose the most helpful and harmless response. 

## Instruction

{{
    "instruction": """{prompt}"""
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models based on their helpfulness and harmlessness. Select the model that produced the most appropriate response. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''


#JUDGES = {"pair_rm": PairRMJudge, "openai": OpenAIPairwiseJudge, "hf": HfPairwiseJudge}
os.environ["LITELLM_API_KEY"] = "sk-A6MWgEmi6WofqK_RPdleew"

if __name__ == "__main__":
    with wandb.init(
        project='huggingface',
        id='icv3r1p5', # yujiagao-carnegie-mellon-university/huggingface/icv3r1p5
        resume="must",
    ) as run:
        parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
        script_args, training_args, model_config = parser.parse_args_and_config()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

        torch_dtype = torch.bfloat16
        quantization_config = get_quantization_config(model_config)
        if quantization_config is not None:
            config_dict = {
                k: v for k, v in quantization_config.to_dict().items() 
                if not k.startswith('_') and k != 'quant_method'
            }
            training_args.quantization_config = config_dict
            print("Training args quantization config:", training_args.quantization_config)
        else:
            training_args.quantization_config = None

        model_kwargs = dict(
            revision=model_config.model_revision,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=training_args.quantization_config,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )

        if training_args.reward_model_path is not None:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                training_args.reward_model_path,
                num_labels=1,
                trust_remote_code=model_config.trust_remote_code,
                **model_kwargs,
            )
            reward_tokenizer = AutoTokenizer.from_pretrained(
                training_args.reward_model_path,
                trust_remote_code=model_config.trust_remote_code,
                truncation=True,
                truncation_side="left",  # since we judge the completion, truncating left is more appropriate
            )
            if reward_tokenizer.pad_token_id is None:
                reward_tokenizer.pad_token = reward_tokenizer.eos_token
        else:
            reward_model = None
            reward_tokenizer = None
        
        if training_args.judge is not None:
            if training_args.judge == 'unsloth':
                judge_bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Enable 4-bit quantization
                    bnb_4bit_compute_dtype='float16',  # Set computation precision (e.g., float16 or float32)
                    bnb_4bit_use_double_quant=True,    # Optional: Double quantization improves precision
                    bnb_4bit_quant_type="nf4"          # Quantization type (e.g., "nf4" or "fp4")
                )
                model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"  # Replace with your model name

                judge_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=judge_bnb_config,  # Apply 4-bit quantization
                    device_map="auto"               # Automatically map model layers to available devices (e.g., GPU)
                )
                judge_tokenizer = AutoTokenizer.from_pretrained(model_name)
                judge = HfPairwiseJudge(judge_model, judge_tokenizer, system_prompt=SG_PAIRWISE_SYSTEM_PROMPT)
            elif training_args.judge == 'gpt-4o-mini':
                judge = OpenAIPairwiseJudge(model="gpt-4o-mini", system_prompt=SG_PAIRWISE_SYSTEM_PROMPT)
        else:
            judge = None

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            padding_side="left",
            trust_remote_code=model_config.trust_remote_code,
            **model_kwargs,
        )
        if tokenizer.chat_template is None:
            tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        for key, value in list(tokenizer.init_kwargs.items()):
            if isinstance(value, torch.dtype):  # 移除 dtype 类型
                print(f"Removing invalid key from tokenizer config: {key}")
                del tokenizer.init_kwargs[key]

        if training_args.reward_model_path is not None:
            for key, value in list(reward_tokenizer.init_kwargs.items()):
                if isinstance(value, torch.dtype):  # 移除 dtype 类型
                    print(f"Removing invalid key from tokenizer config: {key}")
                    del reward_tokenizer.init_kwargs[key]


        dataset = load_dataset(script_args.dataset_name)
        train_dataset = dataset[script_args.dataset_train_split]
        # train_dataset = train_dataset.select(range(80000))
        train_dataset = train_dataset.select(range(30000, 80000))


        # Add these after model loading:
        print("\nModel quantization verification:")
        print(f"Is model quantized: {any('int8' in str(param.dtype) for param in model.parameters())}")
        print(f"Model parameters dtypes: {set(param.dtype for param in model.parameters())}")
        
        
        trainer = OnlineDPOTrainer(
            model=model,
            reward_model=reward_model,
            judge=judge,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer,
            reward_processing_class=reward_tokenizer,
            peft_config=get_peft_config(model_config)
        )



        if training_args.eval_strategy != "no":
            generation_config = GenerationConfig(
                max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
            )
            completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
            trainer.add_callback(completions_callback)

        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)
