import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import transformers


def generate_prompt_with_history(text,history,tokenizer,max_length=2048):
    prompt = "The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n[|Human|]Hello!\n[|AI|]Hi!"
    history = ["\n[|Human|]{}\n[|AI|]{}".format(x[0],x[1]) for x in history]
    history.append("\n[|Human|]{}\n[|AI|]".format(text))
    history_text = ""

    for x in history[::-1]:
        if tokenizer(prompt+history_text+x, return_tensors="pt")['input_ids'].size(-1) <= max_length:
            history_text = x + history_text
            flag = True
    if flag:
        return  prompt+history_text,tokenizer(prompt+history_text, return_tensors="pt")
    else:
        return False

def load_tokenizer_and_model(base_model,adapter_model,load_8bit=False):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            adapter_model,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            adapter_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            adapter_model,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    return tokenizer,model,device


def greedy_search(input_ids: torch.Tensor,
                  model: torch.nn.Module,
                  tokenizer: transformers.PreTrainedTokenizer,
                  stop_words: list,
                  max_length: int,
                  temperature: float = 1.0,
                  top_p: float = 1.0,
                  top_k: int = 25):
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # apply top_k
        #if top_k is not None:
        #    probs_sort1, _ = torch.topk(probs_sort, top_k)
        #    min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
        #    probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)

        yield text
        if any([x in text for x in stop_words]):
            return 

def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False

text='''
你好,请讲一个笑话:
'''
history=[]
base_model='decapoda-research/llama-7b-hf'
adapter_model='project-baize/baize-lora-7B'
tokenizer,model,device = load_tokenizer_and_model(base_model,adapter_model)
inputs = generate_prompt_with_history(text,history,tokenizer,max_length=256)
prompt,inputs=inputs
input_ids = inputs["input_ids"].to(device)
with torch.no_grad():
    for x in greedy_search(input_ids,model,tokenizer,stop_words=["[|Human|]", "[|AI|]"],max_length=256,temperature=1,top_p=1):
        # print(x)
        # print("*"*80)
        if is_stop_word_or_prefix(x,["[|Human|]", "[|AI|]"]) is True:
            if "[|Human|]" in x:
                x = x[:x.index("[|Human|]")].strip()
            if "[|AI|]" in x:
                x = x[:x.index("[|AI|]")].strip() 
            x = x.strip(" ")
# print(prompt)
print('用户：')
print(text)
print('机器：')
print(x)
# print("="*80)
