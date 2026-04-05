# dpo_with_sft_fixed.py
# A complete, runnable version of your "Code1" with the key fixes:
#   1) SFT tokenization: safer prompt/response boundary masking
#   2) SFT collation: pad with tokenizer.pad_token_id (NOT 0)
#   3) SFT dataset returns attention_mask; collator pads it consistently
#   4) Optional debug prints for valid label tokens + loss sanity

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Any
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trl import DPOTrainer, DPOConfig
import wandb
from transformers.integrations import WandbCallback

# Muon optimizer (from the user-provided muon.py)
from muon_v1_conflict import MuonWithAuxAdam, build_muon_param_groups

# ──────────────────────────────────────────────
# Script Arguments
# ──────────────────────────────────────────────

@dataclass
class ScriptArguments:
    """Arguments for DPO + SFT combined training."""

    # DPO
    beta: Optional[float] = field(default=0.01)

    # Model
    model_name_or_path: Optional[str] = field(default="HuggingFaceH4/zephyr-7b-beta")

    # Optimizer / LR
    learning_rate: Optional[float] = field(default=5e-4)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    warmup_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.05)
    optimizer_type: Optional[str] = field(default="adamw_torch")

    # ── Muon optimizer (from muon.py) ─────────────────
    use_muon_optimizer: bool = field(default=True)

    # Adam side (aux optimizer inside MuonWithAuxAdam)
    adam_emb_lr: Optional[float] = field(default=None)
    adam_scalar_lr: Optional[float] = field(default=None)
    adam_betas_muon_side: Tuple[float, float] = field(default=(0.9, 0.95))
    adam_eps_small: float = field(default=1e-8)
    adam_weight_decay: Optional[float] = field(default=None, metadata={"help": "If set, overrides weight_decay for Adam-side groups."})
    adam_max_grad_norm: float = field(default=0.0)
    adam_only_param_regexes: List[str] = field(
    default_factory=lambda: [r"(?:^|\.)(lm_head)(?:\.|$)"]
)
    include_convs_in_muon: bool = field(default=False)

    # Muon side
    muon_momentum: float = field(default=0.95)
    muon_weight_decay: float = field(default=0.0)
    muon_ns_steps: int = field(default=5)
    muon_nesterov: bool = field(default=True)
    muon_use_kimi_scaling: bool = field(default=True)
    muon_kimi_c: float = field(default=0.2)
    muon_eps: float = field(default=1e-7)
    muon_split_mode: str = field(default="v1")  # use Muon v0 (SVD top-k) for split updates
    muon_v0_k: int = field(default=256)
    muon_conflict_check_T: int = field(default=1)
    muon_conflict_path: Optional[str] = field(default=None)  # jsonl; default: <output_dir>/muon_conflict.jsonl
    distributed_broadcast: bool = field(default=False)

    # Batch / grad
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    gradient_checkpointing: Optional[bool] = field(default=True)
    gradient_checkpointing_use_reentrant: Optional[bool] = field(default=False)

    # Sequence lengths
    max_prompt_length: Optional[int] = field(default=512)
    max_length: Optional[int] = field(default=512)

    # Training schedule
    max_steps: Optional[int] = field(default=1000)
    logging_steps: Optional[int] = field(default=50)
    save_steps: Optional[int] = field(default=1000)
    eval_steps: Optional[int] = field(default=200)
    num_train_epochs: Optional[int] = field(default=3)

    # I/O
    output_dir: Optional[str] = field(default="./results")
    log_freq: Optional[int] = field(default=1)
    load_in_4bit: Optional[bool] = field(default=False)  # kept but not used here
    model_dtype: Optional[str] = field(default="bfloat16")

    # Misc
    sanity_check: Optional[bool] = field(default=False)
    report_to: Optional[str] = field(default="wandb")
    ignore_bias_buffers: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=0)

    # DPO with reward-model weighting
    lamb: Optional[float] = field(default=None)
    prefer_safe: bool = field(default=True)
    b_bar: float = field(default=0.0)
    
    # # v1
    # configs.v1_target_param_names = [
    # "model.layers.5.mlp.down_proj.weight",
    # "model.layers.6.mlp.down_proj.weight",
    # "model.layers.7.mlp.down_proj.weight",
    # ]
    # configs.v1_target_steps = [10, 20, 30, 40]
    

    # ── SFT loss ──────────────────────────────────
    gamma: float = field(
        default=0.1,
        metadata={"help": "Weight for SFT loss. total_loss = dpo_loss + gamma * sft_loss"},
    )
    sft_alpaca_ratio: float = field(
        default=0.7,
        metadata={"help": "Fraction of SFT data drawn from Alpaca (rest from GSM8K)."},
    )
    sft_max_seq_length: int = field(
        default=512,
        metadata={"help": "Max token length for SFT sequences."},
    )

    # ── Debug ────────────────────────────────────
    debug_first_sft_batch: bool = field(
        default=False,
        metadata={"help": "If True, print SFT batch stats (valid labels, loss) at step 0."},
    )


# ──────────────────────────────────────────────
# Prompt templates (PKU DPO)
# ──────────────────────────────────────────────

PROMPT_BEGIN: str = "BEGINNING OF CONVERSATION: "
PROMPT_USER: str = "USER: {input} "
PROMPT_ASSISTANT: str = "ASSISTANT:"


# ──────────────────────────────────────────────
# PKU dataset loader
# ──────────────────────────────────────────────

def get_PKU(
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc: int = 24,
    prefer_safe: bool = True,
) -> Tuple[Dataset, Dataset]:
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10K", cache_dir=cache_dir)

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    train_dataset = dataset["train"]
    ds = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]
    original_columns = train_dataset.column_names

    def return_prompt_and_responses(samples) -> Dict[str, List[str]]:
        response_0 = samples["response_0"]
        response_1 = samples["response_1"]
        responses_list = list(zip(response_0, response_1))
        indices = samples["safer_response_id"] if prefer_safe else samples["better_response_id"]
        return {
            "prompt": [
                PROMPT_BEGIN + PROMPT_USER.format(input=q) + PROMPT_ASSISTANT
                for q in samples["prompt"]
            ],
            "chosen": [r[idx] for r, idx in zip(responses_list, indices)],
            "rejected": [r[1 - idx] for r, idx in zip(responses_list, indices)],
        }

    return (
        train_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        ),
        eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        ),
    )


# ──────────────────────────────────────────────
# SFT dataset loader  (Alpaca + GSM8K)
# ──────────────────────────────────────────────

def get_sft_dataset(
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512,
    alpaca_ratio: float = 0.7,
    sanity_check: bool = False,
    num_proc: int = 8,
) -> Dataset:
    """
    Mixed SFT dataset from tatsu-lab/alpaca and gsm8k.

    Key robustness choices:
      - Tokenize full_text once (full_enc).
      - Tokenize prompt alone WITHOUT truncation, to estimate prompt boundary.
      - prompt_len = min(len(prompt_ids), len(full_ids)) to handle truncation.
      - Return attention_mask from tokenizer; pad consistently in collate.
    """

    # ── Alpaca ───────────────────────────────────
    alpaca_raw = load_dataset("tatsu-lab/alpaca", split="train")

    def format_alpaca(ex):
        if ex.get("input", "").strip():
            prompt = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context.\n\n"
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                "### Response:\n"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task.\n\n"
                f"### Instruction:\n{ex['instruction']}\n\n"
                "### Response:\n"
            )
        return {"sft_prompt": prompt, "sft_response": ex["output"]}

    alpaca = alpaca_raw.map(
        format_alpaca,
        num_proc=num_proc,
        remove_columns=alpaca_raw.column_names,
    )

    # ── GSM8K ────────────────────────────────────
    gsm_raw = load_dataset("gsm8k", "main", split="train")

    def format_gsm8k(ex):
        prompt = (
            "Solve the following math problem step by step.\n\n"
            f"### Problem:\n{ex['question']}\n\n"
            "### Solution:\n"
        )
        return {"sft_prompt": prompt, "sft_response": ex["answer"]}

    gsm = gsm_raw.map(
        format_gsm8k,
        num_proc=num_proc,
        remove_columns=gsm_raw.column_names,
    )

    # ── Mix by ratio ─────────────────────────────
    n_alpaca = len(alpaca)
    n_gsm = len(gsm)

    if alpaca_ratio >= 1.0:
        mixed = alpaca
    elif alpaca_ratio <= 0.0:
        mixed = gsm
    else:
        target_gsm = int(n_alpaca * (1 - alpaca_ratio) / alpaca_ratio)
        target_gsm = min(target_gsm, n_gsm)
        target_alpaca = int(target_gsm * alpaca_ratio / (1 - alpaca_ratio))
        target_alpaca = min(target_alpaca, n_alpaca)

        alpaca_sub = alpaca.select(range(target_alpaca))
        gsm_sub = gsm.select(range(target_gsm))
        mixed = concatenate_datasets([alpaca_sub, gsm_sub]).shuffle(seed=42)

    if sanity_check:
        mixed = mixed.select(range(min(len(mixed), 500)))

    # ── Tokenize ─────────────────────────────────
    def tokenise(examples):
        all_input_ids: List[List[int]] = []
        all_labels: List[List[int]] = []
        all_attention_masks: List[List[int]] = []

        for prompt, response in zip(examples["sft_prompt"], examples["sft_response"]):
            full_text = prompt + response + (tokenizer.eos_token or "")

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_seq_length,
                add_special_tokens=True,
            )
            full_ids = full_enc["input_ids"]
            full_attn = full_enc["attention_mask"]

            prompt_ids = tokenizer(
                prompt,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[: len(full_ids)]

            # Ensure lengths match
            if len(labels) != len(full_ids):
                raise ValueError(f"labels/input_ids mismatch: {len(labels)} vs {len(full_ids)}")

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attention_masks.append(full_attn)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_masks,
        }

    mixed = mixed.map(tokenise, batched=True, num_proc=num_proc, remove_columns=mixed.column_names)
    mixed.set_format(type="torch")
    return mixed


# ──────────────────────────────────────────────
# Custom DPO Trainer with SFT loss
# ──────────────────────────────────────────────

class DPOWithSFTTrainer(DPOTrainer):
    """
    total_loss = dpo_loss + gamma * sft_loss

    SFT dataset is cycled independently of DPO dataset.
    """

    def __init__(self, *args, sft_dataset: Dataset, gamma: float = 0.1, debug_first_sft_batch: bool = False, muon_split_mode: str = 'v1', **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)
        self._sft_dataset = sft_dataset
        self._sft_iter = None
        self._debug_first_sft_batch = bool(debug_first_sft_batch)
        self._did_debug_print = False
        self._muon_split_mode = str(muon_split_mode)

        # Make sure tokenizer has a pad token id
        if self.tokenizer.pad_token_id is None:
            # fallback to eos
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_sft_batch(self) -> Dict[str, torch.Tensor]:
        if self._sft_iter is None:
            from torch.utils.data import DataLoader
            loader = DataLoader(
                self._sft_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self._sft_collate_fn,
            )
            self._sft_iter = iter(loader)

        try:
            batch = next(self._sft_iter)
        except StopIteration:
            from torch.utils.data import DataLoader
            loader = DataLoader(
                self._sft_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self._sft_collate_fn,
            )
            self._sft_iter = iter(loader)
            batch = next(self._sft_iter)

        return batch

    def _sft_collate_fn(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Pad {input_ids, labels, attention_mask} to max length in batch.
        IMPORTANT: pad input_ids with tokenizer.pad_token_id (NOT 0).
        """
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = [ex["input_ids"] for ex in examples]
        labels = [ex["labels"] for ex in examples]
        attn = [ex["attention_mask"] for ex in examples]

        max_len = max(len(ids) for ids in input_ids)

        padded_ids, padded_labels, padded_attn = [], [], []
        for ids, lab, am in zip(input_ids, labels, attn):
            pad_len = max_len - len(ids)

            padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)]))
            padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)]))
            padded_attn.append(torch.cat([am, torch.zeros(pad_len, dtype=am.dtype)]))

        return {
            "input_ids": torch.stack(padded_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attn).long(),
        }

    @staticmethod
    def _count_valid_labels(labels: torch.Tensor) -> int:
        return int((labels != -100).sum().item())

    def _compute_dpo_sft_losses(self, model, inputs: Dict[str, Any], **kwargs):
        """Compute and return (dpo_loss, sft_loss)."""
        import inspect
        parent_sig = inspect.signature(super().compute_loss)
        if "num_items_in_batch" in parent_sig.parameters:
            dpo_result = super().compute_loss(
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=kwargs.get("num_items_in_batch"),
            )
        else:
            dpo_result = super().compute_loss(model, inputs, return_outputs=False)

        dpo_loss = dpo_result

        sft_batch = self._get_sft_batch()
        sft_batch = {k: v.to(model.device) for k, v in sft_batch.items()}
        sft_outputs = model(
            input_ids=sft_batch["input_ids"],
            attention_mask=sft_batch["attention_mask"],
            labels=sft_batch["labels"],
        )
        sft_loss = sft_outputs.loss
        return dpo_loss, sft_loss

    def _init_split_accumulators(self):
        if not hasattr(self, "_split_grads_dpo") or self._split_grads_dpo is None:
            self._split_grads_dpo = {}
        if not hasattr(self, "_split_grads_sft") or self._split_grads_sft is None:
            self._split_grads_sft = {}

    def _accumulate_split_grads(self, dpo_grads: dict, sft_grads: dict):
        # Sum across gradient accumulation micro-steps.
        for k, g in dpo_grads.items():
            if g is None:
                continue
            if k not in self._split_grads_dpo:
                self._split_grads_dpo[k] = g
            else:
                self._split_grads_dpo[k].add_(g)
        for k, g in sft_grads.items():
            if g is None:
                continue
            if k not in self._split_grads_sft:
                self._split_grads_sft[k] = g
            else:
                self._split_grads_sft[k].add_(g)

    def training_step(self, model, inputs: Dict[str, Any], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        We need per-loss momentum (DPO vs SFT) conflict.
        Strategy:
        1) backward scaled DPO loss -> snapshot muon grads
        2) backward scaled gamma*SFT loss (accumulates) -> total grads
        3) sft_muon_grad = total_muon_grad - dpo_muon_grad
        4) Let Trainer handle optimizer.step; we store split grads and use a custom optimizer_step().
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        dpo_loss, sft_loss = self._compute_dpo_sft_losses(model, inputs, num_items_in_batch=num_items_in_batch)
        total_loss = dpo_loss + self.gamma * sft_loss

        # gradient accumulation scaling (same as HF Trainer)
        ga = int(max(1, self.args.gradient_accumulation_steps))
        scaled_dpo = dpo_loss / ga
        scaled_sft = (self.gamma * sft_loss) / ga

        self._init_split_accumulators()

        # 1) backward DPO
        self.accelerator.backward(scaled_dpo)

        # snapshot muon grads after DPO backward
        muon_params = []
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for g in getattr(self.optimizer, "param_groups", []):
                if g.get("use_muon", False):
                    muon_params.extend(g["params"])
        dpo_muon = {}
        for p in muon_params:
            if p.grad is None:
                continue
            dpo_muon[id(p)] = p.grad.detach().clone()

        # 2) backward SFT (accumulates into .grad)
        self.accelerator.backward(scaled_sft)

        # total muon grads
        total_muon = {}
        for p in muon_params:
            if p.grad is None:
                continue
            total_muon[id(p)] = p.grad.detach().clone()

        # 3) derive SFT muon grads
        sft_muon = {}
        for pid, gtot in total_muon.items():
            gdpo = dpo_muon.get(pid, None)
            if gdpo is None:
                sft_muon[pid] = gtot
            else:
                sft_muon[pid] = gtot - gdpo

        # 4) accumulate across micro-steps
        self._accumulate_split_grads(dpo_muon, sft_muon)
        
        opt = getattr(self, "optimizer", None)

        # 1) accelerate wrapper: AcceleratedOptimizer has .optimizer
        if hasattr(opt, "optimizer"):
            opt = opt.optimizer

        # 2) accelerate has unwrap_optimizer (有就用)
        if hasattr(self, "accelerator") and hasattr(self.accelerator, "unwrap_optimizer"):
            try:
                opt = self.accelerator.unwrap_optimizer(opt)
            except Exception:
                pass

        # 现在 opt 应该就是 MuonWithAuxAdam
        if not isinstance(opt, MuonWithAuxAdam):
            raise RuntimeError(f"Expected MuonWithAuxAdam, got {type(opt)}")

        opt._pending_split = {
            "forget_grads": self._split_grads_dpo,
            "retain_grads": self._split_grads_sft,
            "mode": "v1",
        }

        # Return *unscaled* loss for logging
        return total_loss.detach()

    def optimizer_step(self, *args, **kwargs):
        # Let HF/Accelerate handle gradient accumulation; only step when sync_gradients=True.
        if hasattr(self, "accelerator") and (not self.accelerator.sync_gradients):
            return

        optimizer = kwargs.get("optimizer", None)
        if optimizer is None:
            # Try to find optimizer in args (HF version differences)
            for a in args:
                if isinstance(a, MuonWithAuxAdam):
                    optimizer = a
                    break
        if optimizer is None:
            optimizer = getattr(self, "optimizer", None)

        if optimizer is None:
            raise RuntimeError("optimizer_step: cannot find optimizer")

        # IMPORTANT: MuonWithAuxAdam.step() is split-only (requires self._pending_split set in training_step)
        optimizer.step()
        return
        def compute_loss(
            self,
            model,
            inputs: Dict[str, Any],
            return_outputs: bool = False,
            **kwargs,  # absorbs num_items_in_batch, etc.
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
            # 1) DPO loss
            import inspect
            parent_sig = inspect.signature(super().compute_loss)
            if "num_items_in_batch" in parent_sig.parameters:
                dpo_result = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=kwargs.get("num_items_in_batch"),
                )
            else:
                dpo_result = super().compute_loss(model, inputs, return_outputs=return_outputs)

            dpo_loss = dpo_result[0] if return_outputs else dpo_result

            # 2) SFT loss
            sft_batch = self._get_sft_batch()
            sft_batch = {k: v.to(model.device) for k, v in sft_batch.items()}

            sft_outputs = model(
                input_ids=sft_batch["input_ids"],
                attention_mask=sft_batch["attention_mask"],
                labels=sft_batch["labels"],
            )
            sft_loss = sft_outputs.loss

            # Optional one-time debug print at step 0
            if self._debug_first_sft_batch and (not self._did_debug_print) and self.state.global_step == 0:
                with torch.no_grad():
                    valid = self._count_valid_labels(sft_batch["labels"])
                    total = int(sft_batch["labels"].numel())
                    print(
                        f"[DEBUG] SFT batch: valid_labels={valid}/{total} "
                        f"({100.0*valid/max(total,1):.2f}%), sft_loss={float(sft_loss.detach().float().cpu()):.4f}"
                    )
                self._did_debug_print = True

            # 3) Combine
            total_loss = dpo_loss + self.gamma * sft_loss
            
            # total_loss = sft_loss

            # Log separately
            if hasattr(self, "log") and self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
                # NOTE: self.log expects plain python numbers
                self.log(
                    {
                        "sft_loss": float(sft_loss.detach().float().cpu()),
                        "dpo_loss": float(dpo_loss.detach().float().cpu()) if torch.is_tensor(dpo_loss) else float(dpo_loss),
                        "gamma": float(self.gamma),
                    }
                )

            if return_outputs:
                return total_loss, dpo_result[1]
            return total_loss


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Disable wandb by default unless user wants it
    # wandb.init(mode="disabled")
    
    
    # Use output_dir folder name as wandb run name
    out_name = os.path.basename(os.path.normpath(script_args.output_dir))
    wandb_run_name = out_name
    
    
    wandb.init(
        project="safety_dpo_sft",
        name=wandb_run_name,
        mode="online",   # online/offline
    )
    

    # dtype
    torch_dtype = torch.float32
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    _ = Accelerator()  # keeps parity with your original; not strictly required

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        # Common decoder-only practice: pad with eos
        tokenizer.pad_token = tokenizer.eos_token

    set_seed(script_args.seed)

    # DPO dataset
    print("script_args.lamb:", script_args.lamb)
    if script_args.lamb is None:
        trainset, evalset = get_PKU(
            sanity_check=script_args.sanity_check,
            prefer_safe=script_args.prefer_safe,
        )
    else:
        # Keep your branch; tokenizer decode expects prompt_ids/response_ids fields etc.
        import json
        with open("PKU-SafeRLHF-30K_train_v3.json", "r") as f:
            train_dataset = json.load(f)
        trainset = Dataset.from_dict(train_dataset)
        ds = trainset.train_test_split(test_size=0.1, seed=42)
        trainset = ds["train"]
        evalset = ds["test"]

        def return_prompt_and_responses(samples):
            reward_scores_0 = torch.tensor(samples["reward_scores_0"])
            reward_scores_1 = torch.tensor(samples["reward_scores_1"])
            safety_scores_0 = -torch.tensor(samples["cost_scores_0"])
            safety_scores_1 = -torch.tensor(samples["cost_scores_1"])

            weight_0s = reward_scores_0 + script_args.lamb * safety_scores_0
            weight_1s = reward_scores_1 + script_args.lamb * safety_scores_1
            probs = torch.sigmoid(weight_1s - weight_0s)
            indices = torch.bernoulli(probs).long().squeeze()

            return {
                "prompt": [tokenizer.decode(p, skip_special_tokens=True) for p in samples["prompt_ids"]],
                "chosen": [
                    tokenizer.decode(samples["response_0_ids"][i], skip_special_tokens=True)
                    if idx == 0
                    else tokenizer.decode(samples["response_1_ids"][i], skip_special_tokens=True)
                    for i, idx in enumerate(indices)
                ],
                "rejected": [
                    tokenizer.decode(samples["response_1_ids"][i], skip_special_tokens=True)
                    if idx == 0
                    else tokenizer.decode(samples["response_0_ids"][i], skip_special_tokens=True)
                    for i, idx in enumerate(indices)
                ],
            }

        trainset = trainset.map(return_prompt_and_responses, batched=True)
        evalset = evalset.map(return_prompt_and_responses, batched=True)

    # SFT dataset
    print(f"Loading SFT dataset (alpaca_ratio={script_args.sft_alpaca_ratio}) ...")
    sft_dataset = get_sft_dataset(
        tokenizer=tokenizer,
        max_seq_length=script_args.sft_max_seq_length,
        alpaca_ratio=script_args.sft_alpaca_ratio,
        sanity_check=script_args.sanity_check,
    )
    print(f"SFT dataset size: {len(sft_dataset)}")

    # DPO config
    dpo_config = DPOConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        eval_steps=script_args.eval_steps,
        report_to=script_args.report_to,
        bf16=(script_args.model_dtype == "bfloat16"),
        fp16=(script_args.model_dtype == "float16"),
        remove_unused_columns=False,
        seed=script_args.seed,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        max_steps=script_args.max_steps,
        run_name=wandb_run_name,
        max_grad_norm=1.0,
    )

    # Build optimizer
    optimizers = None
    if getattr(script_args, "use_muon_optimizer", False):
        # Build Muon param groups (Linear weights -> Muon, others -> Adam)
        cfg = SimpleNamespace(
            # Adam side
            adam_emb_lr=(script_args.adam_emb_lr or script_args.learning_rate),
            adam_scalar_lr=(script_args.adam_scalar_lr or script_args.learning_rate),
            adam_betas_muon_side=script_args.adam_betas_muon_side,
            adam_eps_small=script_args.adam_eps_small,
            adam_weight_decay=(script_args.adam_weight_decay if script_args.adam_weight_decay is not None else script_args.weight_decay),
            adam_max_grad_norm=script_args.adam_max_grad_norm,
            adam_only_param_regexes=script_args.adam_only_param_regexes,
            include_convs_in_muon=script_args.include_convs_in_muon,
            # Muon side
            muon_lr=script_args.learning_rate,
            muon_momentum=script_args.muon_momentum,
            muon_weight_decay=script_args.muon_weight_decay,
            muon_ns_steps=script_args.muon_ns_steps,
            muon_nesterov=script_args.muon_nesterov,
            muon_max_grad_norm=0.0,
            muon_use_kimi_scaling=script_args.muon_use_kimi_scaling,
            muon_kimi_c=script_args.muon_kimi_c,
            muon_eps=script_args.muon_eps,
            distributed_broadcast=script_args.distributed_broadcast,
        V0_K=script_args.muon_v0_k,
            V0_CAP=True,
            CONFLICT_check_T=script_args.muon_conflict_check_T,
            CONFLICT_path=(script_args.muon_conflict_path or os.path.join(script_args.output_dir, 'muon_conflict.jsonl')),
            
        )
        
        cfg.v1_target_param_names = ['model.layers.0.mlp.up_proj.weight', 'model.layers.13.mlp.down_proj.weight', 'model.layers.14.mlp.down_proj.weight', 'model.layers.14.mlp.gate_proj.weight', 'model.layers.14.self_attn.o_proj.weight', 'model.layers.14.self_attn.v_proj.weight', 'model.layers.15.self_attn.q_proj.weight', 'model.layers.16.self_attn.v_proj.weight', 'model.layers.18.mlp.gate_proj.weight', 'model.layers.18.self_attn.o_proj.weight', 'model.layers.18.self_attn.q_proj.weight', 'model.layers.18.self_attn.v_proj.weight', 'model.layers.19.mlp.gate_proj.weight', 'model.layers.19.mlp.up_proj.weight', 'model.layers.19.self_attn.k_proj.weight', 'model.layers.19.self_attn.o_proj.weight', 'model.layers.20.self_attn.v_proj.weight', 'model.layers.23.mlp.gate_proj.weight', 'model.layers.23.self_attn.o_proj.weight', 'model.layers.25.self_attn.k_proj.weight', 'model.layers.25.self_attn.v_proj.weight', 'model.layers.31.mlp.gate_proj.weight']
        cfg.v1_target_steps = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
        # cfg.v1_target_param_regexes = [r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"]

        param_groups = build_muon_param_groups(model, cfg)
        optimizer = MuonWithAuxAdam(param_groups)
        optimizers = (optimizer, None)  # let Trainer create scheduler
    
    # Trainer
    dpo_trainer = DPOWithSFTTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        beta=script_args.beta,
        train_dataset=trainset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        peft_config=None,
        sft_dataset=sft_dataset,
        gamma=script_args.gamma,
        debug_first_sft_batch=script_args.debug_first_sft_batch,
        optimizers=optimizers,
        muon_split_mode=script_args.muon_split_mode,
    )

    dpo_trainer.add_callback(WandbCallback())
    
    # Train
    dpo_trainer.train()

    print("Saving model to:", script_args.output_dir)
    dpo_trainer.save_model(script_args.output_dir)

    final_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    os.makedirs(final_dir, exist_ok=True)
    dpo_trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done. Final checkpoint saved to:", final_dir)


if __name__ == "__main__":
    main()