# -*- coding: utf-8 -*-
import os
import math
import json
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

from cose_dataset import Llama2SummaryDataset

IGNORE_INDEX = -100

# --------------------- Utilities ---------------------
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_llama2_chat(model_name_or_path: str, tokenizer_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path is not None else model_name_or_path,
        use_fast=True,
        cache_dir="/home/huang341/myspace/unlearn_hallucination/llamma-7b-model",
    )
    if tokenizer.pad_token is None:
        # Llama-2 typically has no pad; use eos as pad
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        cache_dir="/home/huang341/myspace/unlearn_hallucination/llamma-7b-model",
    )
    return model, tokenizer


# =========================================================
# ============ Muon core helpers (ported & minimal) =======
# =========================================================
@torch.no_grad()
def _flat2d(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    if x.ndim == 2:
        return x
    elif x.ndim == 4:
        return x.view(len(x), -1)
    else:
        return x.view(x.size(0), -1)


@torch.no_grad()
def _muon_scale(rows: int, cols: int, use_kimi_scaling: bool = False, kimi_c: float = 0.2):
    if use_kimi_scaling:
        return kimi_c * (max(rows, cols) ** 0.5)
    else:
        return (max(1.0, rows / max(cols, 1))) ** 0.5  # sqrt(Out/In)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int, eps: float = 1e-15):
    """
    Quintic Newton–Schulz style orthogonalization (float32 core), then cast back.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    need_cast_back = G.dtype in (torch.float16, torch.bfloat16)
    X = G.to(torch.float32)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    out = X.to(G.dtype) if need_cast_back else X
    return out


@torch.no_grad()
def muon_from_precomputed_M(M: torch.Tensor, *, ns_steps: int = 5,
                            use_kimi_scaling: bool = False, kimi_c: float = 0.2):
    """
    Given a constructed update matrix/tensor M, perform:
    reshape->NS-orthogonalize->shape-aware scale->reshape back.
    """
    M2d = _flat2d(M)  # [rows, cols]
    rows, cols = M2d.shape
    U = zeropower_via_newtonschulz5(M2d, steps=ns_steps)
    scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    update2d = U * scale
    return update2d.view_as(M)


@torch.no_grad()
def special_svd_fusion_update(mbf: torch.Tensor, mbr: torch.Tensor, *, ns_steps: int) -> torch.Tensor:
    """
    Do SVD on mbf/mbr separately; concatenate; orthogonalize; return fused update.
    (float32 SVD for stability)
    """
    mbf32 = mbf.to(torch.float32)
    mbr32 = mbr.to(torch.float32)
    Uf, Sf, Vhf = torch.linalg.svd(mbf32, full_matrices=False)
    Ur, Sr, Vhr = torch.linalg.svd(mbr32, full_matrices=False)
    U_hat  = torch.cat([Ur, Uf], dim=1)   # [m, k_r+k_f]
    Vh_hat = torch.cat([Vhr, Vhf], dim=0) # [k_r+k_f, n]
    V_hat = Vh_hat.mT
    U_hat_orth = zeropower_via_newtonschulz5(U_hat, steps=ns_steps)
    V_hat_orth = zeropower_via_newtonschulz5(V_hat, steps=ns_steps)
    update32 = U_hat_orth @ V_hat_orth.mT
    return update32.to(mbf.dtype)


# =========================================================
# ============== Param grouping (Muon / Adam) =============
# =========================================================
def build_muon_param_groups(model, *, lr: float, betas=(0.9, 0.999), eps=1e-8,
                            adam_weight_decay: float = 0.01,
                            muon_momentum: float = 0.95,
                            muon_weight_decay: float = 0.1,
                            muon_ns_steps: int = 5,
                            muon_nesterov: bool = True,
                            max_grad_norm: float = 1.0,
                            use_kimi_scaling: bool = True,
                            kimi_c: float = 0.2,
                            muon_eps: float = 1e-7):
    """
    简化版分组：
      - Muon组: 所有 nn.Linear.weight (2D) 参数
      - Adam组: 其他（Embedding、bias、LayerNorm/RMSNorm、以及非2D参数）
    """
    muon_params, adam_params = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight") and module.weight is not None:
            muon_params.append(module.weight)

    seen = set(id(p) for p in muon_params)
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        adam_params.append(p)

    muon_group = dict(
        params=sorted(muon_params, key=lambda x: x.numel(), reverse=True),
        use_muon=True,
        lr=lr,
        momentum=muon_momentum,
        weight_decay=muon_weight_decay,
        ns_steps=muon_ns_steps,
        nesterov=muon_nesterov,
        max_grad_norm=max_grad_norm,
        use_kimi_scaling=use_kimi_scaling,
        kimi_c=kimi_c,
        muon_eps=muon_eps,
    )
    adam_group = dict(
        params=list(adam_params),  # keep order
        use_muon=False,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=adam_weight_decay,
        max_grad_norm=max_grad_norm,
    )


    return [muon_group, adam_group]


    # ==== 强制全部走 Adam ====
    # adam_params = [p for p in model.parameters() if p.requires_grad]
    # return [dict(
    #     params=adam_params,
    #     lr=1e-5,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=0.01,
    #     max_grad_norm=1.0,
    #     use_muon=False,      # 明确 false
    # )]





# =========================================================
# ================= Optimizer: Muon + Adam ================
# =========================================================
class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    支持:
      - use_muon 组：读取 p.grad_f_accum / p.grad_r_accum 与 p.grad
         · special 分支（use_special_now=True）：两支动量的 SVD 融合更新
         · 非 special：常规 Muon（由 p.grad 或 Nesterov 形式构造 M）
      - 非 use_muon 组：标准 AdamW
    """
    def __init__(self, param_groups,
                 enable_special_svd: bool = True,
                 special_svd_interval: int = 1,
                 log_interval: int = 100,
                 record_cos_sim: bool = False,
                 cos_sim_interval: int = 10,
                 special_cos_threshold: float = 0.0  # <<< 新增
                  ):
        # defaults
        for g in param_groups:
            has_flag = ("use_muon" in g)
            assert has_flag, "Each param group must specify 'use_muon'."
            g.setdefault("max_grad_norm", 0.0)
            g.setdefault("use_kimi_scaling", True)
            g.setdefault("kimi_c", 0.2)
            if g.get("use_muon", False):
                g["params"] = sorted(list(g["params"]), key=lambda x: x.numel(), reverse=True)
            else:
                g["params"] = list(g["params"])
        super().__init__(param_groups, {})

        # special-SVD trigger & logging
        self._global_step = 0
        self._log_interval = int(log_interval)
        self.enable_special_svd = bool(enable_special_svd)
        self.special_svd_interval = max(1, int(special_svd_interval))

        self.cos_history = {
                "steps_sample": [],
                "cos_M_sample": [],
            }
        self.record_cos_sim = record_cos_sim
        self.cos_sim_interval = max(1, int(cos_sim_interval))
        self.special_cos_threshold = float(special_cos_threshold)  # <<< 新增


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # group-level grad clipping first
        for group in self.param_groups:
            if group.get("max_grad_norm", 0.0) and group["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_grad_norm"])

            if group.get("use_muon", False):
                # ===== Muon side =====
                beta = float(group.get("momentum", 0.95))
                nesterov = bool(group.get("nesterov", True))
                ns_steps = int(group.get("ns_steps", 5))
                lr = float(group["lr"])
                wd = float(group.get("weight_decay", 0.0))
                use_kimi = bool(group.get("use_kimi_scaling", True))
                kimi_c = float(group.get("kimi_c", 0.2))

                for p in group["params"]:
                    # allow cases where p.grad is None but split grads exist
                    gf = getattr(p, "grad_f_accum", None)
                    gr = getattr(p, "grad_r_accum", None)
                    if p.grad is None and gf is None and gr is None:
                        continue

                    # ---- pull split grads (fallback 0) ----
                    if gf is None: gf = torch.zeros_like(p)
                    if gr is None: gr = torch.zeros_like(p)

                    # ---- init 3 momenta ----
                    state = self.state[p]
                    if "momentum_buffer_orig" not in state:
                        state["momentum_buffer_orig"] = torch.zeros_like(p)
                    if "momentum_buffer_f" not in state:
                        state["momentum_buffer_f"] = torch.zeros_like(p)
                    if "momentum_buffer_r" not in state:
                        state["momentum_buffer_r"] = torch.zeros_like(p)
                    mb  = state["momentum_buffer_orig"]
                    mbf = state["momentum_buffer_f"]
                    mbr = state["momentum_buffer_r"]

                    # unified "preheat" three buffers every step
                    if p.grad is not None:
                        mb.mul_(beta).add_(p.grad)
                    if gf is not None:
                        mbf.mul_(beta).add_(gf)
                    if gr is not None:
                        mbr.mul_(beta).add_(gr)

                    # ---- decide use_special_now (cos sign flip) ----
                    # [?]这个逻辑也不太一样

                    param_name = getattr(p, "_param_name", "")
                    is_qkv = (
                        param_name.endswith("q_proj.weight") or
                        param_name.endswith("k_proj.weight") or
                        param_name.endswith("v_proj.weight")
                    )

                    special_candidate = self.enable_special_svd and is_qkv


                    # special_candidate = self.enable_special_svd
                    # if special_candidate and (self._global_step % self.special_svd_interval == 0):
                    if special_candidate:
                        if nesterov:
                            Mbf_dec = gf.to(torch.float32) + mbf.to(torch.float32) * beta
                            Mbr_dec = gr.to(torch.float32) + mbr.to(torch.float32) * beta
                        else:
                            Mbf_dec = mbf.to(torch.float32)
                            Mbr_dec = mbr.to(torch.float32)
                        M1_2d = _flat2d(Mbf_dec)
                        M2_2d = _flat2d(Mbr_dec)
                        eps = 1e-12
                        ip = (M1_2d * M2_2d).sum()
                        n1 = M1_2d.norm(); n2 = M2_2d.norm()
                        cos_dec = (ip / (n1 * n2 + eps)).item()
                        prev_cos = state.get("prev_cos_M_dec", None)
                        
                        
                        # use_special_now = (prev_cos is not None) and (prev_cos > 0.0) and (cos_dec < 0.0)
                        # state["prev_cos_M_dec"] = float(cos_dec)
                    
                        # 不再依赖 prev_cos，直接与阈值比较
                        use_special_now = (cos_dec < self.special_cos_threshold)
                        state["prev_cos_M_dec"] = float(cos_dec)  # 保留记录，日后可视化仍然可用
                    
                    else:
                        use_special_now = False

                    # [!]
                    # use_special_now = False

                    # ---- produce update ----
                    if use_special_now:
                        print("lalala")
                        print(getattr(p, "_param_name", ""))

                        # special-SVD fusion on two branches' decision momentum
                        Mbf = (gf.add(mbf, alpha=beta) if nesterov else mbf)
                        Mbr = (gr.add(mbr, alpha=beta) if nesterov else mbr)
                        M1_2d = _flat2d(Mbf)
                        M2_2d = _flat2d(Mbr)
                        assert M1_2d.ndim == 2 and M2_2d.ndim == 2, "special SVD requires 2D matrices"
                        update_orth = special_svd_fusion_update(M1_2d, M2_2d, ns_steps=ns_steps)
                        rows, cols = M1_2d.shape
                        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi, kimi_c=kimi_c)
                        update = (update_orth * scale).view_as(Mbf)
                    else:
                        
                        # for plot
                        # if self.record_cos_sim:
                        if self.record_cos_sim and (self._global_step % self.cos_sim_interval == 0):
                            if nesterov:
                                Mbf_dec = gf.to(torch.float32) + mbf.to(torch.float32) * beta
                                Mbr_dec = gr.to(torch.float32) + mbr.to(torch.float32) * beta
                            else:
                                print(123)
                                Mbf_dec = mbf.to(torch.float32)
                                Mbr_dec = mbr.to(torch.float32)
                            M1_2d_dec = _flat2d(Mbf_dec)
                            M2_2d_dec = _flat2d(Mbr_dec)
                            eps = 1e-12
                            ip_dec = (M1_2d_dec * M2_2d_dec).sum()
                            n1_dec = M1_2d_dec.norm()
                            n2_dec = M2_2d_dec.norm()
                            cos_M_dec = (ip_dec / (n1_dec * n2_dec + eps)).item()

                            # 计算 g 指标（Frobenius）
                            gff = gf.to(torch.float32)
                            grf = gr.to(torch.float32)
                            ip_g = (gff * grf).sum()
                            ngf = gff.norm()
                            ngr = grf.norm()
                            cos_g = ip_g / (ngf * ngr + eps)


                            
                            state = self.state[p]
                            metrics = state.get("metrics")
                            if metrics is None:
                                metrics = state["metrics"] = {
                                    # —— 采样版（每 10 步一次）——
                                    "steps_sample": [],
                                    "cos_M_sample": [],
                                    "cos_g_sample": [],
                                }
                            metrics["steps_sample"].append(int(self._global_step))
                            metrics["cos_M_sample"].append(float(cos_M_dec))
                            metrics["cos_g_sample"].append(cos_g.detach().cpu().item())
 

                        # normal Muon from orig momentum (Nesterov on p.grad + β·mb)
                        if p.grad is None:
                            M = mb
                        else:
                            M = (p.grad.add(mb, alpha=beta) if nesterov else mb)
                        update = muon_from_precomputed_M(
                            M, ns_steps=ns_steps, use_kimi_scaling=use_kimi, kimi_c=kimi_c
                        )

                    # decoupled weight decay + apply update
                    if wd != 0.0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

            else:
                # [?]参数数值大小对么 ？ 
                # ===== Adam side =====
                lr = float(group["lr"])
                betas = group.get("betas", (0.9, 0.999))
                eps = float(group.get("eps", 1e-8))
                wd  = float(group.get("weight_decay", 0.0))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    grad = p.grad
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
                    bias_c1 = 1 - betas[0] ** state["step"]
                    bias_c2 = 1 - betas[1] ** state["step"]
                    step_update = exp_avg / (exp_avg_sq.sqrt() / math.sqrt(bias_c2) + eps) / (bias_c1)
                    if wd != 0.0:
                        p.mul_(1 - lr * wd)
                    p.add_(step_update, alpha=-lr)

        self._global_step += 1
        return loss


# =========================================================
# ================== Trainer with 3-part loss =============
# =========================================================
class TripleLossTrainer(Trainer):
    def __init__(self, *args, tokenizer: AutoTokenizer = None,
                 max_len: int = 2048,
                 anchor_max: int = 32,
                 anchor_min: int = 3,
                 anchor_back_window: int = 16,
                 lambda_l2: float = 1.0,  # keep
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.anchor_max = anchor_max
        self.anchor_min = anchor_min
        self.anchor_back_window = anchor_back_window
        self.lambda_l2 = lambda_l2

        # ✅ 新增：记录loss曲线
        self.loss_f_history = []
        self.loss_r_history = []

    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)

    # Helper: chat prompt ids for a user message
    def _prompt_ids(self, prompt: str) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        return ids.to(self.model.device)

    # Helper: tokenize response with offsets
    def _resp_tokens_with_offsets(self, text: str):
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        ids = enc.input_ids[0].to(self.model.device)
        offsets = enc.offset_mapping[0].tolist()
        return ids, offsets

    @staticmethod
    def _charspan_to_tokspan(offsets: List[Tuple[int, int]], s: int, e: int) -> Tuple[int, int]:
        # map char span [s,e) to token span [i,j)
        i = 0
        while i < len(offsets) and offsets[i][1] <= s:
            i += 1
        while i > 0 and offsets[i-1][0] >= s:
            i -= 1
        j = i
        while j < len(offsets) and offsets[j][0] < e:
            j += 1
        return i, j

    def _token_norm(self, tid: int) -> str:
        s = self.tokenizer.decode([tid], skip_special_tokens=True).strip()
        s = s.translate(str.maketrans('', '', ',.;:!?"\'`“”‘’()[]{}'))
        return s

    def _equal_seq(self, a_ids: torch.Tensor, b_ids: torch.Tensor) -> bool:
        if a_ids.size(0) != b_ids.size(0):
            return False
        for x, y in zip(a_ids.tolist(), b_ids.tolist()):
            if self._token_norm(x) != self._token_norm(y):
                return False
        return True

    def _find_anchor(self,
                     clean_ids: torch.Tensor,
                     prefix_ids: torch.Tensor,
                     anchor_max: int = None,
                     anchor_min: int = None,
                     back_window: int = None) -> int:
        if anchor_max is None: anchor_max = self.anchor_max
        if anchor_min is None: anchor_min = self.anchor_min
        if back_window is None: back_window = self.anchor_back_window
        Lp = prefix_ids.size(0)
        Lc = clean_ids.size(0)
        if Lp == 0 or Lc == 0:
            return 0
        k_max = min(anchor_max, Lp)
        k_min = max(1, anchor_min)
        for k in range(k_max, k_min - 1, -1):
            start_lo = max(0, Lp - back_window - k)
            start_hi = max(0, Lp - k)
            for s in range(start_lo, start_hi + 1):
                sub = prefix_ids[s:s+k]
                for i in range(0, max(0, Lc - k) + 1):
                    if self._equal_seq(clean_ids[i:i+k], sub):
                        return i + k
        return 0

    def _ce_on_segment(self, prompt_ids: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
        # Trim to max_len budget
        max_resp = max(1, self.max_len - prompt_ids.size(0))
        seg_ids = seg_ids[:max_resp]
        input_ids = torch.cat([prompt_ids, seg_ids], dim=0).unsqueeze(0)
        labels = input_ids.clone()
        labels[0, :prompt_ids.size(0)] = IGNORE_INDEX
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # (1, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction='mean'
        )
        return loss



    def save_loss_plot(self, save_path: str):
        import matplotlib.pyplot as plt

        if len(self.loss_f_history) == 0:
            print("No loss history recorded.")
            return

        plt.figure()
        plt.plot(self.loss_f_history, label="loss_f")
        plt.plot(self.loss_r_history, label="loss_r")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"[Saved] Loss curve to {save_path}")


    def training_step(self, model, inputs):
        """
        修改点：
          - 计算 l1,l2,l3 后，令 loss_r = l1 + l3，loss_f = l2
          - 分别 autograd.grad 得到 grads_r / grads_f，并写入 p.grad_r_accum / p.grad_f_accum（按 GA 缩放并跨 micro-step 累积）
          - 将 p.grad 设置为 (gr + gf) / grad_accum，供 clip 与 Adam 组使用
          - 不在此处 backward，返回 detached 的总损失，外层不会再反传
        """
        model.train()
        prompts:   List[str] = inputs['prompt']
        originals: List[str] = inputs['original_response']
        cleaneds:  List[str] = inputs['cleaned_response']
        spans_list: List[List[Dict[str, Any]]] = inputs.get('ebi_spans', [[] for _ in range(len(prompts))])

        # 一次前传内累计三个分支总和，避免重复构图
        l1_sum = 0.0
        l2_sum = 0.0
        l3_sum = 0.0

        for prompt, orig, clean, spans in zip(prompts, originals, cleaneds, spans_list):
            # 1) prompt ids
            p_ids = self._prompt_ids(prompt)
            # 2) tokenize original & cleaned with offsets
            o_ids, o_offsets = self._resp_tokens_with_offsets(orig)
            c_ids, _ = self._resp_tokens_with_offsets(clean)
            # 3) pick first hallucination span
            if spans and len(spans) > 0 and all(k in spans[0] for k in ('start','end')):
                ch_s = int(spans[0]['start']); ch_e = int(spans[0]['end'])
                i, j = self._charspan_to_tokspan(o_offsets, ch_s, ch_e)
            else:
                i, j = o_ids.size(0), o_ids.size(0)  # no hallucination

            prefix_ids = o_ids[:i]
            halluc_ids = o_ids[i:j]

            # loss_1: predict prefix given prompt
            l1 = self._ce_on_segment(p_ids, prefix_ids) if prefix_ids.numel() > 0 else torch.tensor(0.0, device=o_ids.device)

            # loss_2: negative CE on hallucination conditioned on prompt+prefix
            pp_ids = torch.cat([p_ids, prefix_ids], dim=0)
            l2_pos = self._ce_on_segment(pp_ids, halluc_ids) if halluc_ids.numel() > 0 else torch.tensor(0.0, device=o_ids.device)
            l2 = - self.lambda_l2 * l2_pos

            # loss_3: anchor into cleaned continuation
            x = self._find_anchor(c_ids, prefix_ids)
            prompt_clean = torch.cat([p_ids, c_ids[:x]], dim=0)
            l3 = self._ce_on_segment(prompt_clean, c_ids[x:]) if c_ids.size(0) - x > 0 else torch.tensor(0.0, device=o_ids.device)

            l1_sum = l1_sum + l1
            l2_sum = l2_sum + l2
            l3_sum = l3_sum + l3

        # average over batch
        # [?]需要加权么？
        bs = max(1, len(prompts))
        loss_f = l2_sum / bs
        loss_r = (l1_sum + l3_sum) / bs
        batch_loss = (loss_f + loss_r)

        # ✅ 记录（detach + cpu + float）
        self.loss_f_history.append(loss_f.detach().cpu().item())
        self.loss_r_history.append(loss_r.detach().cpu().item())

        params = [p for p in model.parameters() if p.requires_grad]
        # scale = 1.0 / max(1, self.args.gradient_accumulation_steps)
        scale = 1.0

        grads_f = torch.autograd.grad(loss_f, params, retain_graph=True, allow_unused=True)
        grads_r = torch.autograd.grad(loss_r, params, retain_graph=False, allow_unused=True)


        # [?]这个地方考虑accum 所以你要好好想一下 这个地方 可能不太对

        # 写入参数属性（跨 micro-step 累积），并设置 p.grad 为总和（供裁剪 & Adam 组）
        for p, gf, gr in zip(params, grads_f, grads_r):


            if hasattr(p, "grad_f_accum"):
                delattr(p, "grad_f_accum")
            if hasattr(p, "grad_r_accum"):
                delattr(p, "grad_r_accum")


            if gf is not None:
                gfa = getattr(p, "grad_f_accum", None)
                if gfa is None:
                    setattr(p, "grad_f_accum", gf.detach() * scale)
                else:
                    print(123)
                    setattr(p, "grad_f_accum", gfa + gf.detach() * scale)
            if gr is not None:
                gra = getattr(p, "grad_r_accum", None)
                if gra is None:
                    setattr(p, "grad_r_accum", gr.detach() * scale)
                else:
                    print(123)
                    setattr(p, "grad_r_accum", gra + gr.detach() * scale)

            if (gf is not None) or (gr is not None):
                gf0 = torch.zeros_like(p) if gf is None else gf.detach()
                gr0 = torch.zeros_like(p) if gr is None else gr.detach()
                g_sum = (gf0 + gr0) * scale
                if p.grad is None:
                    p.grad = g_sum
                else:
                    print(123)
                    p.grad = p.grad + g_sum

        # 返回 detached 的 loss，让外层不做 backward
        return batch_loss.detach()


# --------------------- Train entry ---------------------
def train_with_triple_loss(
    model_name: str,
    data_file: str,
    out_dir: str,
    tokenizer_path: str = None,
    batch_size: int = 1,
    epochs: int = 1,
    lr: float = 1e-5,
    max_len: int = 2048,
    anchor_max: int = 32,
    anchor_min: int = 3,
    anchor_back_window: int = 16,
    grad_accum: int = 1,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    train_samples: int = -1,
    lambda_l2: float = 1.0,
    # Muon/Adam configs
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.1,
    muon_ns_steps: int = 5,
    muon_nesterov: int = 1,
    muon_use_kimi_scaling: int = 1,
    muon_kimi_c: float = 0.2,
    enable_special_svd: int = 1,
    special_svd_interval: int = 1,

    record_cos_sim: int = 0,
    cos_sim_interval: int = 10,
    special_cos_threshold: float = 0.0,   # <<< 新增
):
    set_seed(42)
    model, tokenizer = load_llama2_chat(model_name, tokenizer_path)
    # model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
    
    
    # >>> 新增：给所有参数挂上可读的 _param_name，用于日志/导出 <<<
    for name, p in model.named_parameters():
        setattr(p, "_param_name", name)



    dataset = Llama2SummaryDataset(data_file)
    if train_samples is not None and train_samples > 0:
        n = min(int(train_samples), len(dataset))
        train_dataset = Subset(dataset, list(range(n)))
        print(f"[Data] Using only the first {n} samples for training.")
    else:
        train_dataset = dataset

    args = TrainingArguments(
        remove_unused_columns=False,
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,

        # num_train_epochs=epochs,        
        # save_strategy="epoch",
        # save_strategy="steps",
        # save_strategy="no",

        max_steps=300,      
        save_strategy="steps",
        save_steps=100,


        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        report_to="none",
        logging_dir=os.path.join(out_dir, "logs"),
        logging_strategy="steps",
        logging_steps=2,
        gradient_accumulation_steps=grad_accum,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
    )

    # [?]
    # === 构建 Muon / Adam 参数组 ===
    param_groups = build_muon_param_groups(
        model,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_weight_decay=args.weight_decay,
        muon_momentum=muon_momentum,
        muon_weight_decay=muon_weight_decay,
        muon_ns_steps=muon_ns_steps,
        muon_nesterov=bool(muon_nesterov),
        max_grad_norm=args.max_grad_norm,
        use_kimi_scaling=bool(muon_use_kimi_scaling),
        kimi_c=muon_kimi_c,
        muon_eps=1e-7,
    )

    # ===== 新增打印 =====
    # print("\n========== MUON PARAMS ==========")
    # for p in param_groups[0]["params"]:
    #     print(getattr(p, "_param_name", "unknown"))

    # print("\n========== ADAM PARAMS ==========")
    # for p in param_groups[1]["params"]:
    #     print(getattr(p, "_param_name", "unknown"))

    # print("Muon params:", len(param_groups[0]["params"]))
    # print("Adam params:", len(param_groups[1]["params"]))

    optimizer = MuonWithAuxAdam(
        param_groups,
        enable_special_svd=bool(enable_special_svd),
        special_svd_interval=int(special_svd_interval),
        log_interval=100,
        record_cos_sim=bool(record_cos_sim),
        cos_sim_interval=int(cos_sim_interval),
        special_cos_threshold=float(special_cos_threshold),  # <<< 新增
    )

    # === Scheduler ===
    num_update_steps_per_epoch = math.ceil(
        len(train_dataset) / args.per_device_train_batch_size / args.gradient_accumulation_steps
    )
    num_training_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = TripleLossTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        data_collator=dataset.collate_fn,
        optimizers=(optimizer, lr_scheduler),
        max_len=max_len,
        anchor_max=anchor_max,
        anchor_min=anchor_min,
        anchor_back_window=anchor_back_window,
        lambda_l2=lambda_l2,
    )
    model.config.use_cache = False
    trainer.train()

    # ✅ 保存loss曲线
    loss_plot_path = os.path.join(out_dir, "loss_curve.png")
    trainer.save_loss_plot(loss_plot_path)


    # cos_save_path = os.path.join(out_dir, "cos_M_history.json")
    # with open(cos_save_path, "w") as f:
    #     json.dump(optimizer.cos_history, f, indent=2)
    # print(f"[Saved] cos_M history to {cos_save_path}")


    # —— 保存 per-param 指标 ——  (放在保存 loss 之后)
    metrics_out = {}
    for group in optimizer.param_groups:
        if group.get("use_muon", False):
            for p in group["params"]:
                name = getattr(p, "_param_name", f"param_{id(p)}")
                st = optimizer.state[p]
                m = st.get("metrics", None)
                if m is not None:
                    metrics_out[name] = {
                        "steps_sample": m.get("steps_sample", []),
                        "cos_M_sample": m.get("cos_M_sample", []),
                        "cos_g_sample": m.get("cos_g_sample", []),
                    }
    metrics_file = os.path.join(out_dir, "muon_dual_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print(f"Muon dual metrics saved to {metrics_file}")








if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Finetune Llama-2-7b-chat with triple loss (with Muon split-grads).')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--data', type=str, default='/home/huang341/myspace/unlearn_hallucination/dataset/train/llama2_Summary_single.json')
    parser.add_argument('--out', type=str, default='/home/huang341/myspace/unlearn_hallucination/temp_1/train_sft_model/batch_1_unlearn_1.5_hybrid_muon')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--anchor_max', type=int, default=32)
    parser.add_argument('--anchor_min', type=int, default=3)
    parser.add_argument('--anchor_back_window', type=int, default=16)
    parser.add_argument('--lambda_l2', type=float, default=2.0)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--train_samples', type=int, default=100)

    # Muon/Adam 配置
    parser.add_argument('--muon_momentum', type=float, default=0.95)
    parser.add_argument('--muon_weight_decay', type=float, default=0.1)
    parser.add_argument('--muon_ns_steps', type=int, default=5)
    parser.add_argument('--muon_nesterov', type=int, default=1)
    parser.add_argument('--muon_use_kimi_scaling', type=int, default=1)
    parser.add_argument('--muon_kimi_c', type=float, default=0.2)

    
    parser.add_argument('--enable_special_svd', type=int, default=1)
    parser.add_argument('--special_svd_interval', type=int, default=0)


    parser.add_argument('--record_cos_sim', type=int, default=0)
    parser.add_argument('--cos_sim_interval', type=int, default=10)
    parser.add_argument('--special_cos_threshold', type=float, default=-0.3,
                    help='当 cos_dec 低于该阈值时触发 special SVD（默认 0.0）')

    args = parser.parse_args()
    train_with_triple_loss(
        model_name=args.model,
        data_file=args.data,
        out_dir=args.out,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_len=args.max_len,
        anchor_max=args.anchor_max,
        anchor_min=args.anchor_min,
        anchor_back_window=args.anchor_back_window,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        train_samples=args.train_samples,
        lambda_l2=args.lambda_l2,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_ns_steps=args.muon_ns_steps,
        muon_nesterov=args.muon_nesterov,
        muon_use_kimi_scaling=args.muon_use_kimi_scaling,
        muon_kimi_c=args.muon_kimi_c,
        enable_special_svd=args.enable_special_svd,
        special_svd_interval=args.special_svd_interval,

        record_cos_sim=args.record_cos_sim,
        cos_sim_interval=args.cos_sim_interval,  # ✅ 新增
        special_cos_threshold=args.special_cos_threshold,  # <<< 新增
    )
