from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle

for model_id in ["google/gemma-2-9b-it", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "allenai/OLMo-7B-Instruct-hf", "facebook/opt-6.7b", "tiiuae/falcon-7b-instruct", "mistralai/Mistral-Nemo-Instruct-2407"]:
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    with open('../prompts/prompts.pkl', 'rb') as file:
        prompts = pickle.load(file)
        
        
    # Format chat
    inputs = []

    for input_list in prompts:

        system_message = "Continue the following dialogue without any additional explanations or format changes.\n\n"
        messages = []

        # Determine the appropriate starting role for the rest of the messages
        roles = ["assistant", "user"]
        starting_role_index = 1

        # Transform the input list into the desired format
        messages.extend([{"role": roles[(i + starting_role_index) % 2], "content": message} for i, message in enumerate(input_list)])

        messages[0]['content'] = system_message+messages[0]['content']

        inputs.append(messages)
        
    # Truncate very long articles
    #prompts = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(p)[:1024]) for p in prompts]
        
    if "opt" not in model_id and "falcon" not in model_id:
        prompts = tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)

    else:
        prompts = []
        inputs[0][0]['content'] = inputs[0][0]['content'].replace(system_message, '')
        for i in inputs:
            text = ""
            for j in i:
                text += j['role']+': '+j['content'].strip()+'\n'
            prompts.append(text)
        

    f_out = open("../outputs/"+model_id.split("/")[-1]+".txt", 'w')

    batch_size = 32
    
    if "gemma" in model_id or "Qwen" in model_id:
        batch_size = 16
        
    

    for i in range(0, len(prompts), batch_size):

        inputs = tokenizer(prompts[i:i+batch_size], return_tensors="pt", padding=True).to(model.device)

        output_sequences = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.02)

        outputs = tokenizer.batch_decode(output_sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        for output in outputs:
            f_out.write(output.replace('\n', '<newline>').replace('\r', '<newline>').replace('\v', '<newline>') + '\n')