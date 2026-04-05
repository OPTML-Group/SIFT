#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用本地/远程 Llama-2-Chat 对给定数据集(每条含 prompt)批量生成 response，
再调用 GPT 对 (prompt, response) 进行幻觉检测（span-level，字符起止位置）。

输出：JSONL，每行一条样本，包含：
{
  "id": int,
  "task_type": str,
  "prompt": str,
  "model_name": str,
  "response": str,
  "judge_model": str,
  "judge": {
      "hallucinated": bool,
      "spans": [
          {"start": int, "end": int, "text": str,
           "category": "contradiction|unsupported|subjective_exaggeration|speculation",
           "explanation": str}
      ],
      "notes": str
  }
}
支持 --resume 断点续跑；支持限并发与指数回退重试。
"""

import os
import json
import argparse
import time
import math
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- OpenAI (或 Azure OpenAI) ---
# pip install openai>=1.0.0
from openai import OpenAI, AsyncOpenAI
from openai import RateLimitError, APIConnectionError, APITimeoutError, BadRequestError

import asyncio
from asyncio import Semaphore
from tqdm import tqdm


import random, numpy as np



# ========== Llama-2 Chat 加载 ==========
def load_llama2_chat(model_name_or_path: str, tokenizer_path: str = None, cache_dir: Optional[str] = None):
    """
    与用户给出的加载函数一致，仅将 cache_dir 参数化，避免硬编码。
    # """
    # tokenizer = AutoTokenizer.from_pretrained(
    #     tokenizer_path if tokenizer_path is not None else model_name_or_path,
    #     use_fast=True,
    #     cache_dir=cache_dir,
    # )
    # if tokenizer.pad_token is None:
    #     # Llama-2 通常没有 PAD，用 EOS 代替
    #     tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name_or_path,
    #     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    #     device_map="auto",
    #     cache_dir=cache_dir,
    # )



    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path is not None else model_name_or_path,
        use_fast=True,
        # cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        # Llama-2 通常没有 PAD，用 EOS 代替
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        # cache_dir=cache_dir,
    )


    return model, tokenizer


# ========== 生成部分 ==========
def format_llama2_chat_prompt(raw_prompt: str) -> str:
    """
    对 Llama-2-Chat 的最简 [INST] 包裹；你的数据中 prompt 已自包含“任务+原文”，
    因此不再额外注入 system 指令，避免信息偏移。
    """
    p = raw_prompt.strip()
    return f"[INST] {p} [/INST]"

@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> List[str]:


    outs = []
    device = model.device
    for rec in tqdm(records, desc="Generating", total=len(records)):
        prompt = format_llama2_chat_prompt(rec["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,

        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        # 生成文本中有时会包含 prompt，尽量抽取回答段。如果你希望保留全量，可去掉以下切分。
        # 这里按 [/INST] 后的内容作为回答（若找不到，则原样返回）
        marker = "[/INST]"
        if marker in text:
            resp = text.split(marker, 1)[-1].strip()
        else:
            resp = text.strip()
        outs.append(resp)
    return outs


# ========== GPT 评审部分 ==========
JUDGE_SYSTEM_PROMPT = (
    "You are an expert hallucination evaluator. You will be given a source "
    "context and a generated response. Your task is to identify all "
    "hallucinated spans in the response -- text that is either:\n\n"
    "1. **Baseless Info**: Information not substantiated by or inferable from "
    "the source context.\n"
    "2. **Conflict**: Information that directly contradicts the source context.\n\n"
    "Output a JSON object with the following structure:\n"
    '{"hallucination list": ["span1", "span2", ...]}\n\n'
    'If there are no hallucinations, output: {"hallucination list": []}\n\n'
    "Rules:\n"
    "- Each span should be the exact text from the response that is hallucinated.\n"
    "- Prefer longer contiguous spans over many small fragments.\n"
    "- Only flag claims of fact. Do not flag stylistic choices, reasonable "
    "inferences, or standard phrasing.\n"
    "- Be conservative: when in doubt, do NOT flag."
)

def build_messages_for_judge(source: str, answer: str) -> List[Dict[str, str]]:
    user_prompt = f"""SOURCE:
{source}

ANSWER:
{answer}

Task: Identify hallucinated spans in ANSWER w.r.t. SOURCE. Return JSON per the schema."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


async def judge_one(
    client: AsyncOpenAI,
    gpt_model: str,
    source: str,
    answer: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    sem: Optional[Semaphore] = None,
) -> Dict[str, Any]:
    backoff = 2.0
    attempt = 0
    messages = build_messages_for_judge(source, answer)

    async def _call():
        resp = await client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content)

    while True:
        try:
            if sem:
                async with sem:
                    return await _call()
            else:
                return await _call()
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            if attempt >= max_retries:
                return {"hallucinated": False, "spans": [], "notes": f"error: {type(e).__name__}: {str(e)}"}
            await asyncio.sleep(min(backoff ** attempt, 60.0))
            attempt += 1
        except BadRequestError as e:
            # 通常为超长或格式问题，直接返回空
            return {"hallucinated": False, "spans": [], "notes": f"bad_request: {str(e)}"}
        except Exception as e:
            if attempt >= max_retries:
                return {"hallucinated": False, "spans": [], "notes": f"exception: {type(e).__name__}: {str(e)}"}
            await asyncio.sleep(min(backoff ** attempt, 60.0))
            attempt += 1


async def run_judges(
    records: List[Dict[str, Any]],
    responses: List[str],
    gpt_model: str,
    concurrency: int = 16,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    # client = AsyncOpenAI()

    client = AsyncOpenAI(
    api_key=""
)


    sem = Semaphore(concurrency)
    tasks = []
    for rec, ans in zip(records, responses):
        tasks.append(judge_one(
            client=client,
            gpt_model=gpt_model,
            source=rec["prompt"],
            answer=ans,
            temperature=temperature,
            sem=sem
        ))
    results = await asyncio.gather(*tasks)
    return results


# ========== 主流程 ==========
def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 断言字段存在
    for i, d in enumerate(data):
        if "prompt" not in d:
            raise ValueError(f"Record {i} missing 'prompt'.")
    return data



def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # 数据与输出
    # p.add_argument("--input", type=str, default="/home/huang341/myspace/unlearn_hallucination/dataset/test/llama2_Summary_single.json", help="输入数据（见上传文件）")
    p.add_argument("--input", type=str, default="/home/huang341/myspace/unlearn_hallucination/dataset/train/llama2_Summary_single.json", help="输入数据（见上传文件）")

    p.add_argument("--out", type=str, default="llama2_outputs_judged_unlearn_1.jsonl", help="输出 JSONL 路径")
    p.add_argument("--resume", action="store_true", help="若文件存在则断点续跑")
    # Llama2 生成
    # p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Llama-2-Chat 路径或HF仓库名，如 meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--model", type=str, default="/home/huang341/myspace/unlearn_hallucination/temp_1/sft_model/batch_1_unlearn_1.5/checkpoint-23", help="Llama-2-Chat 路径或HF仓库名，如 meta-llama/Llama-2-7b-chat-hf")



    p.add_argument("--tokenizer_path", type=str, default=None, help="可选的 tokenizer 路径")
    p.add_argument("--cache_dir", type=str, default="/home/huang341/myspace/unlearn_hallucination/llamma-7b-model", help="HF 缓存目录")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--gen_temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--no_sample", action="store_true", help="关闭采样（即贪心生成）")
    # GPT 评审
    # p.add_argument("--gpt_model", type=str, default="gpt-4o-mini", help="用于评审的 GPT 模型名")
    p.add_argument("--gpt_model", type=str, default="gpt-5.2", help="用于评审的 GPT 模型名")

    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max_judge", type=int, default=100, help="仅处理  后N条， -1为全部")

    p.add_argument("--seed", type=int, default=42, help="统一随机种子（生成侧）")
    p.add_argument("--deterministic", action="store_false", help="尽量开启确定性算子")


    return p.parse_args()


# 新增：根据 judge 结构判断是否幻觉
def _pred_is_halu(judge: dict) -> bool:
    if isinstance(judge, dict) and "hallucination list" in judge:
        try:
            return len(judge["hallucination list"]) > 0
        except Exception:
            return False
    # 兼容其它结构（如果后续换了判定器）
    if isinstance(judge, dict) and "hallucinated" in judge:
        return bool(judge.get("hallucinated"))
    return False


def main():
    args = parse_args()

    # 加载数据
    records = load_dataset(args.input)
    # if args.max_judge is not None and args.max_judge > 0:
    #     records = records[:args.max_judge]

    if args.max_judge is not None and args.max_judge > 0:
        n = min(int(args.max_judge), len(records))
        records = records[-n:]
        print(f"[Data] Using the last {n} samples.")

    # 读取已有结果做断点续跑
    done_ids = set()
    mode = "w"
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done_ids.add(item.get("id"))
                except Exception:
                    continue
        mode = "a"
        print(f"[Resume] 已存在 {len(done_ids)} 条，将跳过这些样本。")

    # 待处理样本与索引映射
    pending_indices = [i for i in range(len(records)) if i not in done_ids]
    if not pending_indices:
        print("没有需要继续处理的样本。")
        return




    # 统一固定随机 & 确定性
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass


    # 加载 Llama-2 并生成
    model, tokenizer = load_llama2_chat(
        model_name_or_path=args.model,
        tokenizer_path=args.tokenizer_path,
        cache_dir=args.cache_dir
    )

    gen_records = [records[i] for i in pending_indices]


    responses = generate_responses(
        model, tokenizer, gen_records,
        max_new_tokens=args.max_new_tokens,
        temperature=args.gen_temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample
    )

    # 调用 GPT 评审
    loop = asyncio.get_event_loop()
    judge_results = loop.run_until_complete(
        run_judges(
            gen_records, responses,
            gpt_model=args.gpt_model,
            concurrency=args.concurrency,
            temperature=args.judge_temperature
        )
    )

    # 写结果
    halu_cnt = 0
    with open(args.out, mode, encoding="utf-8") as f:
        for local_idx, resp, judge in zip(pending_indices, responses, judge_results):
            rec = records[local_idx]
            out = {
                "id": local_idx,
                "task_type": rec.get("task_type", ""),
                "prompt": rec["prompt"],
                "model_name": args.model,
                "response": resp,
                "judge_model": args.gpt_model,
                "judge": judge
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            
            if _pred_is_halu(judge):  # 使用上面的工具函数
                halu_cnt += 1


    print(f"完成：共写入 {len(pending_indices)} 条结果 -> {args.out}")
    
    total = len(pending_indices)
    ratio = (halu_cnt / total) * 100.0 if total > 0 else 0.0
    print(f"[Summary] hallucination = {halu_cnt}/{total} ({ratio:.2f}%)")



if __name__ == "__main__":
    main()