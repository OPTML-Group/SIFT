import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch.nn as nn

random.seed(0)

################################
##### Activation functions #####
################################

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    


def forward_with_cache_list(model, inputs, module_list, no_grad=True):
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None
    
    hook_handles = [module.register_forward_hook(hook) for module in module_list]
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    [h.remove() for h in hook_handles]

    return cache


def forward_add_perturbed_cache(model, inputs, module_lists, perturbations, no_grad=True):
    cache = []
    perturbations = perturbations
    def add_hook(module, input, output):
        if perturbations.get(module) is not None:
            if isinstance(output, tuple):
               output[0].data = output[0] + perturbations[module]
            else:
                output.data = output + perturbations[module]
        else:
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)

    hook_handles = [module.register_forward_hook(add_hook) for module in module_lists]
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    [h.remove() for h in hook_handles]

    return cache[0]


#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            from datasets import load_dataset
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            for line in open(f"../files/data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )


import json

def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []

        # -------------------------
        # 1) wikitext (HF)
        # -------------------------
        if name == "wikitext":
            from datasets import load_dataset
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                txt = str(x.get("text", ""))
                if min_len < len(txt) <= max_len:
                    data.append(txt)

        # -------------------------
        # 2) cyber (HF)
        # -------------------------
        elif name == "cyber-forget-corpus":
            from datasets import load_dataset
            ds = load_dataset("cais/wmdp-cyber-forget-corpus")
            # pick an available split robustly
            split = "train"
            raw_data = ds[split]
            for x in raw_data:
                data.append(str(x))

        # -------------------------
        # 3) local jsonl
        # -------------------------
        else:
            path = f"/egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl"
            
            for line in open(path, "r"):
                if "bio" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))

        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )


import torch
from torch.optim import AdamW
from typing import Iterable


class SAMAdamW(AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, rho=0.05, gamma=1.0):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        # if len(self.param_groups) > 1:
        #     raise ValueError("Not supported")
        self.rho = rho
        self.gamma = gamma


    @torch.no_grad()
    def step(self, closure) -> torch.Tensor:
        # print("This is SAM!")
        closure_forget = torch.enable_grad()(closure[0])
        closure_retain = torch.enable_grad()(closure[1])

        self.zero_grad() 
        closure_forget().detach()
        # print(f"Forget loss: {loss_forget}")
        for group in self.param_groups:
            grads = []
            params_with_grads = []
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)

            if len(grads) == 0:
                continue

            device = grads[0].device
            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            # 计算 epsilon = rho * grad / ||grad||
            epsilon = []
            for g in grads:
                eps = g * (self.rho / grad_norm.to(g.device))
                epsilon.append(eps)

            # ---------- 第 2 步: 给模型加扰动 ----------
            torch._foreach_add_(params_with_grads, epsilon)

            # ---------- 第 3 步: 在扰动模型上再次调用 closure_forget ----------
            self.zero_grad()
            loss_forget = closure_forget()  # forward + backward on perturbed model
            forget_perturbed_grads = [
                p.grad.clone().detach() if p.grad is not None else None
                for p in params_with_grads
            ]

            torch._foreach_sub_(params_with_grads, epsilon)
            self.zero_grad()
            loss_retain = closure_retain()  # forward + backward on original model
            retain_grads = [
                p.grad.clone().detach() if p.grad is not None else None
                for p in params_with_grads
            ]

            for i, p in enumerate(params_with_grads):
                fg = forget_perturbed_grads[i]
                rg = self.gamma * retain_grads[i]
                p.grad.copy_(fg + rg)

        # clip_grad_norm_(params_with_grads, 1.0)
        super().step()
        loss = loss_forget + self.gamma * loss_retain
        return loss



def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1).mean()
    return loss