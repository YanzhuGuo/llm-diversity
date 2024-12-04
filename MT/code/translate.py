from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

for model_id in ["google/gemma-2-9b-it", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "allenai/OLMo-7B-Instruct-hf", "facebook/opt-6.7b", "tiiuae/falcon-7b-instruct", "mistralai/Mistral-Nemo-Instruct-2407"]:
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    with open('../prompts/prompts.txt', 'r') as f:
        prompts = f.readlines()
        
    # Truncate very long articles
    prompts = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(p)[:1024]) for p in prompts]
        
    if "opt" not in model_id and "falcon" not in model_id:
        messages = [[{"role": "user", "content": "You are a professional translator. You always show the translated version, without any additional explanations or format changes.\n\nTranslate from French to English:\n\n"+p}] for p in prompts]
        prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    else:
        prompts = ["French: "+p+"\nEnglish: " for p in prompts]
        

    f_out = open("../outputs/"+model_id.split("/")[-1]+".txt", 'w')

    batch_size = 32
    
    if 'Qwen' in model_id:
        batch_size = 16

    for i in range(0, len(prompts), batch_size):

        inputs = tokenizer(prompts[i:i+batch_size], return_tensors="pt", padding=True).to(model.device)

        output_sequences = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.02)

        outputs = tokenizer.batch_decode(output_sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        for output in outputs:
            f_out.write(output.replace('\n', '<newline>').replace('\r', '<newline>').replace('\v', '<newline>') + '\n')