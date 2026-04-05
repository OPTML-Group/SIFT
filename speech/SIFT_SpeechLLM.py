# temp1_muon.py
# -------------------------------------------------------------
# Finetuning ChatGLM with Muon optimizer (single machine)
# Dual-loss momentum: m_f, m_r -> O_f, O_r via Newton-Schulz,
# final Muon update uses O_f + O_r.
#
# Sources and design references:
# - Official Muon README: use Muon for hidden matrix weights, AdamW for embeddings/head/bias/gain.
# - muon(1).py implementation: Newton-Schulz iteration in bf16; Muon/AdamW step logic.
# - Muon scaling report: add weight decay to Muon; shape-aware update scaling; distributed notes (we use single-device here).
# -------------------------------------------------------------

from transformers.optimization import get_scheduler
import re
import torch.distributed as dist
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"


import math
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


import torch.nn.functional as F
import transformers
from torch.utils.data import SequentialSampler
from transformers import Trainer, AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# from dataset import DefaultDataset
# from esnli_dataset import DefaultDataset
from cose_dataset import DefaultDataset


from transformers import TrainerCallback, TrainerControl, TrainerState
import argparse
import json

from torch.utils.data import SequentialSampler, Subset  # NEW


# ======================
# Muon core (adapted from muon(1).py)
# ======================



# ---- helpers: parse special indices & load layer list ----
def parse_special_indices(s: str):
    """
    解析类似 "[1]"、"[3 5]"、"1,3,5" 的输入为整数列表（1-based）。
    允许逗号或空格分隔；忽略空串；去重保序。
    """
    import re
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    tokens = re.split(r"[,\s]+", s.strip())
    out = []
    for t in tokens:
        if t:
            try:
                out.append(int(t))
            except ValueError:
                raise ValueError(f"Invalid index token: {t}")
    # 去重保序
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            uniq.append(i); seen.add(i)
    return uniq

def load_layer_list(txt_path: str):
    """
    读取 layer.txt，每行一个全名；自动剔除空行与首尾空白。
    返回: [name1, name2, ...]
    """
    names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names



@torch.no_grad() 
def special_svd_fusion_update(mbf: torch.Tensor, mbr: torch.Tensor, *, ns_steps: int) -> torch.Tensor:
    """对动量缓冲 mbf/mbr 分别做 SVD，按描述拼接并正交化，返回最终 update。"""
    mbf32 = mbf.to(torch.float32)
    mbr32 = mbr.to(torch.float32)
    Uf, Sf, Vhf = torch.linalg.svd(mbf32, full_matrices=False)
    Ur, Sr, Vhr = torch.linalg.svd(mbr32, full_matrices=False)
    U_hat  = torch.cat([Ur, Uf], dim=1)          # [m, k_r + k_f]
    Vh_hat = torch.cat([Vhr, Vhf], dim=0)        # [k_r + k_f, n]
    V_hat  = Vh_hat.mT                           # [n, k_r + k_f]
    U_hat_orth = zeropower_via_newtonschulz5(U_hat, steps=ns_steps)
    V_hat_orth = zeropower_via_newtonschulz5(V_hat, steps=ns_steps)
    update32 = U_hat_orth @ V_hat_orth.mT        # [m, n]
    return update32.to(mbf.dtype)


def _flatten_to_2d(x: torch.Tensor) -> torch.Tensor:
    """
    将张量扁平化到 2D 矩阵：
    - 如果 x.ndim >= 2，返回 [x.size(0), -1]
    - 否则（理论上不会出现在 Muon 组），做保护性处理为 [1, -1]
    """
    if x.ndim >= 2:
        return x.reshape(x.size(0), -1)
    else:
        return x.reshape(1, -1)
    



@torch.no_grad()
def _svd_interaction_metric(mbf: torch.Tensor, mbr: torch.Tensor) -> torch.Tensor:
    """
    计算 || (U^T U - I) Σ (V^T V - I) ||_1 ，其中
    U = [U_r, U_f], V^T = [V_r^T; V_f^T], Σ = block_diag(Σ_r, Σ_f)
    返回一个标量张量（device 与 dtype 设为 float32）。
    """
    # 扁平化到 2D
    A_f = _flatten_to_2d(mbf)
    A_r = _flatten_to_2d(mbr)

    # SVD 需要更稳的float32
    A_f32 = A_f.to(torch.float32)
    A_r32 = A_r.to(torch.float32)

    # economy SVD: U(m,k), S(k,), Vh(k,n)；更利于拼接
    # torch.linalg.svd 在CUDA上支持 float32
    Uf, Sf, Vhf = torch.linalg.svd(A_f32, full_matrices=False)
    Ur, Sr, Vhr = torch.linalg.svd(A_r32, full_matrices=False)

    # 构造帽子矩阵
    # U_hat: [Ur, Uf]  (水平拼接，维度 m x (k_r + k_f))
    U_hat = torch.cat([Ur, Uf], dim=1)

    # V_hat^T: [V_r^T; V_f^T]  (纵向拼接，维度 (k_r + k_f) x n)
    Vh_hat = torch.cat([Vhr, Vhf], dim=0)  # 这是 V^T

    # Σ_hat: block_diag(diag(Sr), diag(Sf))
    Sr_diag = torch.diag(Sr)
    Sf_diag = torch.diag(Sf)
    Sigma_hat = torch.block_diag(Sr_diag, Sf_diag)

    k_total = Sigma_hat.size(0)
    I = torch.eye(k_total, dtype=torch.float32, device=Sigma_hat.device)

    # (U^T U - I): U_hat.mT @ U_hat -> k_total x k_total
    UU_minus_I = (U_hat.mT @ U_hat) - I

    # (V^T V - I): Vh_hat @ Vh_hat.mT -> k_total x k_total  (注意：Vh是V^T)
    VV_minus_I = (Vh_hat @ Vh_hat.mT) - I

    # M = (U^T U - I) Σ (V^T V - I)
    M = UU_minus_I @ Sigma_hat @ VV_minus_I

    return M.abs().sum()





def save_config(training_args: transformers.TrainingArguments, muon_cfg, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    def make_serializable(obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)
    args_dict = {k: make_serializable(v) for k, v in vars(training_args).items()}
    muon_dict = {k: make_serializable(v) for k, v in vars(muon_cfg).items()}
    full_config = {
        "training_args": args_dict,
        "muon_config": muon_dict
    }
    file_path = os.path.join(out_dir, "config.json")
    import json
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=4, ensure_ascii=False)
    print(f"Full training config saved to {file_path}")




def _dist_world_size():
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

def _dist_rank():
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

@torch.no_grad()
def _flat2d(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    if x.ndim == 2:
        return x
    elif x.ndim == 4:
        return x.view(len(x), -1)  # [out, in*kH*kW]
    else:
        return x.view(x.size(0), -1)

@torch.no_grad()
def lrmuon_update(param, grad, state, *, lr: float, momentum: float, nesterov: bool,
                  rank_k: int, refresh_T: int, ns_steps: int, weight_decay: float,
                  weight_norm_stabilize: bool = False,
                  use_kimi_scaling: bool = False, kimi_c: float = 0.2,
                  muon_eps: float = 1e-15):
    if grad is None:
        return  # <— SKIP on None grad

    # momentum
    if 'momentum_buffer' not in state:
        state['momentum_buffer'] = torch.zeros_like(grad)
    buf = state['momentum_buffer']
    buf.mul_(momentum).add_(grad)
    M = grad.add(buf, alpha=momentum) if nesterov else buf

    # decoupled wd (before step)
    if weight_decay != 0.0:
        param.mul_(1 - lr * weight_decay)

    # optional (keep default False)
    if weight_norm_stabilize:
        denom = param.data.norm() + 1e-12
        param.data.mul_((param.data.numel() ** 0.5) / denom)

    # 2D
    M2d = _flat2d(M)
    rows, cols = M2d.shape
    k_eff = int(max(1, min(rank_k, rows, cols)))  # <— safer

    # cache P (top-k left singular vectors)
    if 'lrmuon_step' not in state:
        state['lrmuon_step'] = 0
    need_refresh = (
        ('lrmuon_P' not in state) or
        (state['lrmuon_step'] % int(refresh_T) == 0) or
        (state.get('lrmuon_shape') != (rows, cols)) or
        (state.get('lrmuon_k') != k_eff)
    )
    if need_refresh:
        # ensure SVD runs on same device in float32
        U, S, Vh = torch.linalg.svd(M2d.to(torch.float32), full_matrices=False)
        P = U[:, :k_eff].T.contiguous().to(device=M2d.device, dtype=M2d.dtype)  # (k, rows)
        state['lrmuon_P'] = P
        state['lrmuon_k'] = k_eff
        state['lrmuon_shape'] = (rows, cols)

    state['lrmuon_step'] += 1
    P = state['lrmuon_P']

    # low-rank muon: project → NS → back-project
    X = P @ M2d
    Y = zeropower_via_newtonschulz5(X, steps=ns_steps, eps=muon_eps).to(P.dtype)
    # optional scale in subspace
    scale = _muon_scale(k_eff, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    Y = Y * scale

    update2d = P.mT @ Y
    update = update2d.view_as(M)
    param.add_(update, alpha=-lr)

def _muon_scale(rows, cols, use_kimi_scaling: bool = False, kimi_c: float = 0.2):
    if use_kimi_scaling:
        return kimi_c * (max(rows, cols) ** 0.5)
    else:
        return (max(1.0, rows / max(cols, 1))) ** 0.5  # sqrt(Out/In)

@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps: int, eps: float = 1e-15):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
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
def muon_update(grad, momentum, *,
                beta=0.95, ns_steps=5, nesterov=True,
                use_kimi_scaling: bool = False, kimi_c: float = 0.2):
    if grad is None:
        # return None
        return None, None 

    # 1) momentum
    momentum.mul_(beta).add_(grad)
    M = grad.add(momentum, alpha=beta) if nesterov else momentum

    # 2) 2D
    M2d = _flat2d(M)
    rows, cols = M2d.shape

    # 3) orthogonalize
    U = zeropower_via_newtonschulz5(M2d, steps=ns_steps)

    # 4) scale
    scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    update2d = U * scale
    # return update2d.view_as(M)
    return update2d.view_as(M), M2d  # <— 修改：连同 M2d 返回





@torch.no_grad()
def muon_from_precomputed_M(M: torch.Tensor, *, ns_steps=5,
                            use_kimi_scaling: bool = False, kimi_c: float = 0.2):
    """
    给定已经构造好的矩阵/张量 M（例如 Nesterov: grad + beta * momentum，或纯 momentum），
    只做：2D 重排 -> Newton-Schulz 正交化 -> 形状自适应缩放 -> reshape 回原状。
    不在此函数内做任何动量更新。
    """
    M2d = _flat2d(M)                     # [rows, cols]
    rows, cols = M2d.shape
    U = zeropower_via_newtonschulz5(M2d, steps=ns_steps)  # 正交化
    scale = _muon_scale(rows, cols,
                        use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    update2d = U * scale
    return update2d.view_as(M)



def build_muon_param_groups(model, configs):

    # === NEW: 为每个参数挂上其 name，便于记录和导出 ===
    for name, p in model.named_parameters():
        if not hasattr(p, "_param_name"):
            setattr(p, "_param_name", name)

    # --- enforce exclusivity with a seen set (by object id) ---
    seen = set()
    adam_only_regexes = getattr(configs, "adam_only_param_regexes", [r"(?:^|\.)(lm_head)(?:\.|$)"])
    include_convs = bool(getattr(configs, "include_convs_in_muon", False))

    # collect adam-only by name
    adam_only_patterns = [re.compile(p) for p in adam_only_regexes]
    adam_only_ids = set()
    for name, p in model.named_parameters():
        if any(rx.search(name) for rx in adam_only_patterns):
            adam_only_ids.add(id(p))

    # 2) Embeddings (token/pos)
    embed_params = []
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            p = m.weight
            if id(p) not in seen:
                embed_params.append(p)
                seen.add(id(p))
                adam_only_ids.add(id(p))

    # 3) Scalars: ndim < 2,     # bias / scalars
    scalar_params = []
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        if p.ndim < 2 or id(p) in adam_only_ids:
            scalar_params.append(p)
            seen.add(id(p))

    # collect conv weights (optional muon)
    conv_weight_ids = set()
    conv_types = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    for m in model.modules():
        if isinstance(m, conv_types) and hasattr(m, "weight"):
            conv_weight_ids.add(id(m.weight))

    # Linear weights → Muon
    muon_params = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and hasattr(m, "weight"):
            p = m.weight
            if id(p) not in seen and id(p) not in adam_only_ids:
                muon_params.append(p); seen.add(id(p))
            # bias stays in scalars (already handled)

    # 其余 2D+（非 Linear.weight）：默认走 Adam（或可加更多分组）
    other_2d_params = []
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        if p.ndim >= 2:
            # Conv 默认不上 Muon
            if (id(p) in conv_weight_ids) and (not include_convs):
                other_2d_params.append(p); seen.add(id(p))
            else:
                # 若允许 conv 参与 muon，可追加到 muon_params
                if include_convs:
                    muon_params.append(p); seen.add(id(p))
                else:
                    other_2d_params.append(p); seen.add(id(p))
    
    # AdamW side groups
    adam_groups = [
        dict(params=embed_params,
             lr=configs.adam_emb_lr,
             betas=configs.adam_betas_muon_side,
             eps=configs.adam_eps_small,
             weight_decay=configs.adam_weight_decay,
             max_grad_norm=configs.adam_max_grad_norm,
             use_muon=False),  # <- adam 组保持 use_muon=False
        dict(params=scalar_params + other_2d_params,
             lr=configs.adam_scalar_lr,
             betas=configs.adam_betas_muon_side,
             eps=configs.adam_eps_small,
             weight_decay=0.0,
             max_grad_norm=configs.adam_max_grad_norm,
             use_muon=False),
    ]

    # 选择 Muon 或 Low-Rank-Muon 作为“隐藏矩阵”组
    if getattr(configs, "use_low_rank_muon", False):  # [ADDED]
        lrmuon_group = dict(
            params=muon_params,
            lr=configs.lrmuon_lr,
            momentum=configs.lrmuon_momentum,
            nesterov=configs.lrmuon_nesterov,
            rank_k=configs.lrmuon_rank_k,
            refresh_T=configs.lrmuon_refresh_T,
            ns_steps=configs.lrmuon_ns_steps,
            weight_decay=configs.lrmuon_weight_decay,
            weight_norm_stabilize=configs.lrmuon_weight_norm_stabilize,
            max_grad_norm=configs.lrmuon_max_grad_norm,
            use_kimi_scaling=bool(configs.muon_use_kimi_scaling),
            kimi_c=configs.muon_kimi_c,
            muon_eps=configs.muon_eps,
            distributed_broadcast=bool(configs.distributed_broadcast),
            use_lrmuon=True,
        )
        return [*adam_groups, lrmuon_group]
    else:
        muon_group = dict(
            params=muon_params,
            lr=configs.muon_lr,
            momentum=configs.muon_momentum,
            weight_decay=configs.muon_weight_decay,
            ns_steps=getattr(configs, "muon_ns_steps", 5),     # [ADDED] 可选
            nesterov=getattr(configs, "muon_nesterov", True),  # [ADDED] 可选
            max_grad_norm=getattr(configs, "muon_max_grad_norm", 1.0),
            use_kimi_scaling=bool(getattr(configs, "muon_use_kimi_scaling", False)),
            kimi_c=getattr(configs, "muon_kimi_c", 0.2),
            muon_eps=getattr(configs, "muon_eps", 1e-7),
            distributed_broadcast=bool(getattr(configs, "distributed_broadcast", False)),
            use_muon=True,
        )
        return [*adam_groups, muon_group]

class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups, metric_file_path=None,  metric_interval: int = 0, param_name_map: dict = None, special_svd_steps: int = 5,log_interval: int = 5 , special_param_names: set = None, 
        enable_svd_metric: bool = False,
        enable_special_svd: bool = False,
        enable_cos_metric: bool = False,

        special_svd_interval: int = 1,

        special_cos_threshold: float = 0.0,

        ):
        for group in param_groups:
            has_flag = ("use_muon" in group) or ("use_lrmuon" in group)
            assert has_flag, "Each param group must specify 'use_muon' or 'use_lrmuon' (or set use_muon=False for Adam)."

            # unify safe defaults
            group.setdefault("max_grad_norm", 0.0)
            group.setdefault("distributed_broadcast", False)  # <— default OFF
            group.setdefault("use_kimi_scaling", False)
            group.setdefault("kimi_c", 0.2)

            if group.get("use_muon", False) or group.get("use_lrmuon", False):
                # sort by numel, not shape tuple
                group["params"] = sorted(group["params"], key=lambda x: x.numel(), reverse=True)
            else:
                group["params"] = list(group["params"])  # keep order

        super().__init__(param_groups, {})

        # 用于做cos similarity 统计
        self._global_step = 0       # 优化器的全局 step 计数，从 0 开始
        self._log_interval = int(log_interval)     # 每 10 个 step 记录一次采样指标

        # --- 新增：SVD度量写盘控制 ---
        self.metric_file_path = metric_file_path  # e.g., ".../svd_momentum_metric.jsonl"

        
        self._metric_step = 0  # optimizer本地step计数
        self.metric_interval   = max(1, int(metric_interval))
        self.param_name_map    = param_name_map or {}

        self.special_svd_steps = int(special_svd_steps)  #新增

        
        # === 新增：开关保存 ===
        self.enable_svd_metric = bool(enable_svd_metric)
        self.enable_special_svd = bool(enable_special_svd)
        self.enable_cos_metric  = bool(enable_cos_metric)

        self.special_svd_interval = max(1, int(special_svd_interval))

        self.special_cos_threshold = float(special_cos_threshold)

        
        # ---- special names selection: prefer explicit set; else fallback to regex ----
        if special_param_names is not None and len(special_param_names) > 0:
            # 显式集合优先
            self._special_param_names = set(special_param_names)
            print(f"[Muon] Special (explicit) params selected: {len(self._special_param_names)}")

        else:
            print("error!!!")
            ###########################
            self._special_regexes = [
                re.compile(r'^transformer\.encoder\.layers\.\d+\.self_attention\.dense\.weight$'),
                re.compile(r'^transformer\.encoder\.layers\.\d+\.mlp\.dense_h_to_4h\.weight$'),
            ]
            names = list(self.param_name_map.values())
            self._special_param_names = {name for name in names if any(rx.match(name) for rx in self._special_regexes)}
            # 可选：打印统计
            print(f"[Muon] Special (regex) params selected: {len(self._special_param_names)}")


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ws = _dist_world_size()
        rk = _dist_rank()

        for group in self.param_groups:
            # ---- (A) optional grad clipping per group BEFORE momentum ----
            if group.get("max_grad_norm", 0.0) and group["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_grad_norm"])

            # ---- (B) route by group type ----
            if group.get("use_muon", False):
                params = group["params"]
                for p in params:
                    # 似乎Muon 分支不应该依赖 p.grad 是否存在
                    if p.grad is None:
                        continue  # <— skip None grad
                    
                    # —— 取名、gf/gr 等 ——
                    pname = self.param_name_map.get(id(p), "")
                    is_special = pname in self._special_param_names

                    # —— 从参数属性取出分裂梯度（可能为 None），并兜底到 zeros —— 
                    gf = getattr(p, "grad_f_accum", None)
                    gr = getattr(p, "grad_r_accum", None)
                    if gf is None and gr is None:
                        # 如果分裂梯度都缺失，则继续用 p.grad 驱动 orig 动量
                        pass
                    if gf is None: gf = torch.zeros_like(p, device=p.device, dtype=p.dtype)
                    if gr is None: gr = torch.zeros_like(p, device=p.device, dtype=p.dtype)

                    # —— 初始化三套动量 —— 
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

                    beta     = group["momentum"]
                    nesterov = group.get("nesterov", True)

                    # —— 统一：每一步都“预热”三套动量（保持时间一致性）——
                    if p.grad is not None:
                        mb.mul_(beta).add_(p.grad)
                    if gf is not None:
                        mbf.mul_(beta).add_(gf)
                    if gr is not None:
                        mbr.mul_(beta).add_(gr)
                    # ---------------------------------------------------------------------------------------
                    # --- 新增：基于 cos_M_dec 的“符号翻转 (+→−)”触发 ---
                    use_nesterov = group.get("nesterov", True)
                    beta = float(group["momentum"])
                    if use_nesterov:
                        Mbf_dec = gf.to(torch.float32) + mbf.to(torch.float32) * beta
                        Mbr_dec = gr.to(torch.float32) + mbr.to(torch.float32) * beta
                    else:
                        Mbf_dec = mbf.to(torch.float32)
                        Mbr_dec = mbr.to(torch.float32)
                    M1_2d_dec = _flat2d(Mbf_dec)
                    M2_2d_dec = _flat2d(Mbr_dec)
                    # 计算 cos(M_dec)
                    eps = 1e-12
                    n1_dec = M1_2d_dec.norm()
                    n2_dec = M2_2d_dec.norm()
                    if n1_dec.item() < 1e-20 or n2_dec.item() < 1e-20:
                        # 极早期或极端情况下范数趋近 0，避免误触发：设为 1.0（不触发）
                        cos_M_dec = 1.0
                    else:
                        ip_dec   = (M1_2d_dec * M2_2d_dec).sum()
                        cos_M_dec = (ip_dec / (n1_dec * n2_dec + eps)).item()

                    # 仅由“是否开启”和“是否小于阈值”决定是否使用 special now
                    use_special_now = bool(self.enable_special_svd) and (cos_M_dec < self.special_cos_threshold)

                    # 可选：记录最近一次 cos，便于调试与日志分析
                    state["last_cos_M_dec"] = float(cos_M_dec)


                    # # 仅当候选 special（开关开 + 命中 special 参数）时才计算 cos，避免无谓开销
                    # special_candidate = (self.enable_special_svd and is_special)
                    # if special_candidate:
                    #     # 说明：此时 mbf/mbr 已在上方“预热”为 mbf = β·mbf_old + gf
                    #     # 决策用的 M（不改 state）：Nesterov -> gf + β·mbf；否则 -> mbf
                    #     if use_nesterov:
                    #         Mbf_dec = gf.to(torch.float32) + mbf.to(torch.float32) * beta
                    #         Mbr_dec = gr.to(torch.float32) + mbr.to(torch.float32) * beta
                    #     else:
                    #         Mbf_dec = mbf.to(torch.float32)
                    #         Mbr_dec = mbr.to(torch.float32)
                    #     M1_2d_dec = _flat2d(Mbf_dec)
                    #     M2_2d_dec = _flat2d(Mbr_dec)
                    #     eps = 1e-12
                    #     ip_dec = (M1_2d_dec * M2_2d_dec).sum()
                    #     n1_dec = M1_2d_dec.norm()
                    #     n2_dec = M2_2d_dec.norm()
                    #     cos_M_dec = (ip_dec / (n1_dec * n2_dec + eps)).item()
                    #     prev_cos = state.get("prev_cos_M_dec", None)
                    #     use_special_now = (prev_cos is not None) and (prev_cos > 0.0) and (cos_M_dec < 0.0)
                    #     # 记录本步 cos，供下一步比较
                    #     state["prev_cos_M_dec"] = float(cos_M_dec)
                    # else:
                    #     use_special_now = False


                    # —— 构造 M 并产生 update —— 
                    if use_special_now:
                        print("lalala")
                        print("self._global_step: ", self._global_step)
                        # print("cos similarity: ", state["prev_cos_M_dec"])
                        
                        # 注意：Nesterov 形式 M = g + beta * m，非 Nesterov 就是纯 m
                        Mbf = (gf.add(mbf, alpha=beta) if nesterov else mbf)
                        Mbr = (gr.add(mbr, alpha=beta) if nesterov else mbr)

                        # 统一按 2D 展平，再送入 SVD 融合（函数内部做拼接与正交化）
                        M1_2d = _flat2d(Mbf)
                        M2_2d = _flat2d(Mbr)

                        # special_svd_fusion_update 期望 2D；确保维度正确
                        assert M1_2d.ndim == 2 and M2_2d.ndim == 2, "special SVD fusion expects 2D matrices"
                        update_orth = special_svd_fusion_update(
                            M1_2d, M2_2d, ns_steps=int(group.get("ns_steps", 5))
                        )

                        rows, cols = M1_2d.shape
                        scale = _muon_scale(rows, cols,
                                            use_kimi_scaling=group.get("use_kimi_scaling", False),
                                            kimi_c=group.get("kimi_c", 0.2))
                        update = (update_orth * scale).view_as(Mbf)   # 2D -> 原状（此处 Mbf 与 p 同形状）

                        # —— SVD 交互度量：仅在 special 分支计算并写盘 —— 
                        if self.enable_svd_metric and (self._metric_step % self.metric_interval == 0):
                            try:
                                metric_value = _svd_interaction_metric(M1_2d, M2_2d)  # 已是 2D
                                if self.metric_file_path is not None:
                                    rec = {
                                        "step": int(self._metric_step),
                                        "param_name": self.param_name_map.get(id(p), f"param_{id(p)}"),
                                        "shape": list(p.shape),
                                        "value": float(metric_value.item()),
                                    }
                                    with open(self.metric_file_path, "a") as f:
                                        f.write(json.dumps(rec) + "\n")
                            except Exception:
                                pass

                    else:
                        # normal 分支：用“orig 动量”的 M
                        # Nesterov: M = p.grad + beta * mb；否则 M = mb
                        if p.grad is None:
                            # 理论上不应发生（前面已经预热了 mb），仍做保护
                            continue
                        M = (p.grad.add(mb, alpha=beta) if nesterov else mb)

                        # 直接从预构造的 M 产生更新（不再在这里更新动量！）
                        update = muon_from_precomputed_M(
                            M, ns_steps=int(group.get("ns_steps", 5)),
                            use_kimi_scaling=group.get("use_kimi_scaling", False),
                            kimi_c=group.get("kimi_c", 0.2)
                        )

                    # —— 权重衰减 + 应用更新 —— 
                    if group.get("weight_decay", 0.0) != 0.0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

                # optional param broadcast (disabled by default)
                if ws > 1 and group.get("distributed_broadcast", False):
                    for p in params:
                        dist.broadcast(p.data, src=0)


            else:  # Adam side
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    upd = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                      state["step"], group["betas"], group["eps"])
                    if group.get("weight_decay", 0.0) != 0.0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(upd, alpha=-group["lr"])

        
        self._metric_step += 1
        self._global_step += 1  
        return loss


class Configs:
    # ---- Adam 侧 ----
    rmsnorm_weight_decay: float = 0.1
    adam_emb_lr: float = 1e-5
    adam_scalar_lr: float = 1e-5
    adam_betas_muon_side = (0.9, 0.999)
    adam_eps_small: float = 1e-8
    adam_weight_decay: float = 0.01
    adam_max_grad_norm: float = 1.0
    include_convs_in_muon: bool = False
    adam_only_param_regexes = [r"(?:^|\.)output_layer(?:\.$)"]
    # ---- Muon 组 ----
    muon_lr: float = 1e-5
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    muon_max_grad_norm: float = 1.0
    muon_use_kimi_scaling: bool = True
    muon_kimi_c: float = 0.2
    muon_eps: float = 1e-7
    distributed_broadcast: bool = False
    use_low_rank_muon: bool = False

# ======================
# Callback & Utilities (same as your original)
# ======================

class SaveAfterNEpochsCallback(TrainerCallback):
    """前 n 个 epoch 不保存，之后每个 epoch 都保存"""
    def __init__(self, n: int = 6):
        self.n = n
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch is not None and state.epoch >= self.n:
            control.should_save = True
        return control

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_model_and_tokenizer(model_dir: str, tokenizer_dir: str = None):
    import shutil
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/glm-9b-voice-model")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    model = AutoModel.from_pretrained(
        model_dir,
        device_map="auto",
        # torch_dtype=torch.bfloat16,   # bf16: NS iteration is stable in bf16
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir is not None else model_dir,
        trust_remote_code=True,
    )
    print("tokenizer.pad_token: ", tokenizer.pad_token)
    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
    # print("tokenizer.eos_token: ", tokenizer.eos_token)
    return model, tokenizer


# ======================
# Custom Trainer: Dual-loss gradients accumulation for Muon
# ======================

class DualLossMuonTrainer(Trainer):
    """
    Trainer that computes l_f and l_r separately and accumulates their grads
    into per-parameter attributes:
      - p.grad_f_accum
      - p.grad_r_accum
    Meanwhile, p.grad is set to their SUM (scaled by gradient_accumulation_steps),
    so HF's gradient clipping and scheduler work as usual.  ????

    The optimizer must be SingleDeviceDualMuonWithAuxAdam to consume split grads.
    """

    def __init__(self, *args, lambda_r=1.0, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_r = float(lambda_r)  # if you want to reweight l_r later
        self.tokenizer = tokenizer
        self.loss_log = {"loss": [], "l_f": [], "l_r": []}

    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)

    def training_step(self, model, inputs):
        """
        Compute l_f and l_r, take autograd.grad for each,
        accumulate into p.grad_f_accum / p.grad_r_accum (scaled by GA steps),
        and set p.grad = (grad_f + grad_r)/GA for clipping.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
            labels = inputs.get("labels", input_ids.clone()).to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs.logits.to(labels.device)

            shifted_labels = labels[..., 1:].contiguous()
            shifted_logits = logits[..., :-1, :].contiguous()

 
            # pad_id = self.tokenizer.pad_token_id
            # audio_id0 = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
            # valid = shifted_labels != pad_id
            # audio_mask = (shifted_labels >= audio_id0) & valid
            # text_mask  = (shifted_labels <  audio_id0) & valid

            ignore_index = -100
            valid = shifted_labels != ignore_index
            audio_id0 = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
            audio_mask = (shifted_labels >= audio_id0) & valid
            text_mask  = (shifted_labels <  audio_id0) & valid



            if text_mask.any():
                l_f = F.cross_entropy(
                    shifted_logits[text_mask], shifted_labels[text_mask], reduction="mean"
                )
            else:
                l_f = torch.tensor(0.0, device=labels.device)

            if audio_mask.any():
                l_r = F.cross_entropy(
                    shifted_logits[audio_mask], shifted_labels[audio_mask], reduction="mean"
                )
            else:
                l_r = torch.tensor(0.0, device=labels.device)

        # Token-weighted combination (for logging only)
        n_text = text_mask.sum()
        n_audio = audio_mask.sum()
        total_tokens = max(1, (n_text + n_audio).item())  # avoid div by 0
        loss = (l_f * n_text + l_r * n_audio) / total_tokens

        # === Dual-loss gradients (functional) ===
        params = [p for p in model.parameters() if p.requires_grad]

        # Scale by gradient_accumulation_steps so that HF outer loop can step correctly
        scale = 1.0 / max(1, self.args.gradient_accumulation_steps)

        grads_f = torch.autograd.grad(l_f, params, retain_graph=True, allow_unused=True)
        grads_r = torch.autograd.grad(l_r, params, retain_graph=False, allow_unused=True)


        w_f = float(n_text.item()) / float(total_tokens)
        w_r = float(n_audio.item()) / float(total_tokens)

        # Accumulate into per-parameter slots; set p.grad = sum for clipping
        for p, gf, gr in zip(params, grads_f, grads_r):

            if hasattr(p, "grad_f_accum"):
                delattr(p, "grad_f_accum")
            if hasattr(p, "grad_r_accum"):
                delattr(p, "grad_r_accum")

            # if gf is not None:
            #     gfa = getattr(p, "grad_f_accum", None)
            #     if gfa is None:
            #         setattr(p, "grad_f_accum", gf.detach() * scale)
            #     else:
            #         print(123)
            #         setattr(p, "grad_f_accum", gfa + gf.detach() * scale)
            # if gr is not None:
            #     gra = getattr(p, "grad_r_accum", None)
            #     if gra is None:
            #         setattr(p, "grad_r_accum", gr.detach() * scale)
            #     else:
            #         print(456)
            #         setattr(p, "grad_r_accum", gra + gr.detach() * scale)

            # 给 Muon 侧：把按 token 比例加权过的“平均梯度”写入累积槽
            if gf is not None:
                gfa = getattr(p, "grad_f_accum", None)
                if gfa is None:
                    setattr(p, "grad_f_accum", gf.detach() * w_f * scale)
                else:
                    setattr(p, "grad_f_accum", gfa + gf.detach() * w_f * scale)

            if gr is not None:
                gra = getattr(p, "grad_r_accum", None)
                if gra is None:
                    setattr(p, "grad_r_accum", gr.detach() * w_r * scale)
                else:
                    setattr(p, "grad_r_accum", gra + gr.detach() * w_r * scale)

            # set p.grad (sum) for clipping and for Adam groups
            if (gf is not None) or (gr is not None):
                device = gf.device
                n_text = torch.as_tensor(n_text, dtype=gf.dtype, device=device)
                n_audio = torch.as_tensor(n_audio, dtype=gf.dtype, device=device)
                total_tokens = torch.as_tensor(total_tokens, dtype=gf.dtype, device=device)

                g_sum = ((gf.detach() * n_text + gr.detach() * n_audio) / total_tokens) * scale
                # should add clip here ?
                if p.grad is None:
                    p.grad = g_sum
                else:
                    print("error!")
                    p.grad = p.grad + g_sum

        # Logging
        self.loss_log["loss"].append((l_f + l_r).detach().cpu().item())
        self.loss_log["l_f"].append(l_f.detach().cpu().item())
        self.loss_log["l_r"].append(l_r.detach().cpu().item())

        return loss.detach()

# ======================
# Pretraining (finetuning) function
# ======================

def pretrain(
    model_dir: str,
    data_file: str,
    out_dir: str,
    tokenizer_dir: str = None,
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate: float = 1e-5,  # scheduler LR; optimizer groups set internally
    max_len: int = 512,
    resume_from_checkpoint: bool = False,
    lambda_r: float = 1.0,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,       # Trainer's global WD (we still set WD in optimizer groups)
    max_grad_norm: float = 1.0,

    muon_weight_decay: float = 0.1,
    rmsnorm_weight_decay: float = 0.1,
    muon_kimi_c: float = 0.2,
    muon_use_kimi_scaling: int = 1,

    # === 新增：SVD/日志相关参数传入 optimizer ===
    metric_interval: int = 10000,
    special_svd_steps: int = -1,
    log_interval: int = 10,

    # NEW:
    special_layers_file: str = "layer.txt",
    special_layers_idx: str = "",
    # metric 开关 
    enable_special_svd: int = 0,
    enable_cos_metric: int = 0,
    enable_svd_metric: int = 0,

    special_svd_interval: int = 1,

    train_samples: int = -1,                      # NEW


):
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_dir, tokenizer_dir)
    dataset = DefaultDataset(data_file, tokenizer=tokenizer, max_len=max_len)

    
    # --- NEW: 先取 collate_fn，再按需做子集 ---
    # collate_fn = dataset.get_collate_fn()  # 保留原数据集提供的 collator
    if train_samples is not None and int(train_samples) > 0:
        n = min(int(train_samples), len(dataset))
        train_dataset = Subset(dataset, list(range(n)))
        print(f"[Data] Using only the first {n} samples for training.")
    else:
        train_dataset = dataset




    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,      # used by scheduler only
        
        
        max_steps=200,      
        save_strategy="no",
        # save_steps=25,

        lr_scheduler_type="cosine",

        # bf16=True,
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        # 关键：复现性相关
        seed=42,
        data_seed=42,

        report_to="tensorboard",
        logging_dir=f"{out_dir}/logs",
        logging_strategy="steps",
        logging_steps=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,    
        warmup_ratio=warmup_ratio,
    )


    # === 构造 Muon 的配置并建立优化器 ===
    cfg = Configs()
    cfg.adam_emb_lr = training_args.learning_rate
    cfg.adam_scalar_lr = training_args.learning_rate
    cfg.adam_weight_decay = training_args.weight_decay
    cfg.adam_max_grad_norm = training_args.max_grad_norm
    cfg.muon_lr = training_args.learning_rate
    cfg.muon_max_grad_norm = training_args.max_grad_norm

    # 新增：将 CLI 的 WD & Kimi 参数写入 cfg
    cfg.muon_weight_decay = muon_weight_decay
    cfg.rmsnorm_weight_decay = rmsnorm_weight_decay
    cfg.muon_kimi_c = muon_kimi_c
    cfg.muon_use_kimi_scaling = bool(muon_use_kimi_scaling)

    param_groups = build_muon_param_groups(model, cfg)

    #  构造 id(p) -> name 的映射
    name_map = {id(p): name for name, p in model.named_parameters()}
    metric_path = os.path.join(out_dir, "svd_momentum_metric.jsonl")

    # ---- NEW: special layers by indices from layer.txt ----
    special_param_names = set()
    try:
        if special_layers_idx and special_layers_idx.strip():
            layer_list = load_layer_list(special_layers_file)
            idx_list = parse_special_indices(special_layers_idx)  # 1-based
            for i in idx_list:
                if i < 1 or i > len(layer_list):
                    raise IndexError(f"special index {i} out of range [1..{len(layer_list)}]")
                special_param_names.add(layer_list[i - 1])
            print(f"[Muon] Will use SPECIAL-SVD for {len(special_param_names)} params "
                  f"(by indices from {special_layers_file}):")
            for nm in sorted(special_param_names)[:10]:
                print("   -", nm)
            if len(special_param_names) > 10:
                print(f"   ... (+{len(special_param_names)-10} more)")
    except Exception as e:
        print(f"[WARN] special layers selection failed: {e}; fallback to regex.")

    optimizer = MuonWithAuxAdam(
        param_groups,
        metric_file_path=metric_path,
        metric_interval=metric_interval,           # 这里可改为你想要的步长
        param_name_map=name_map,
        special_svd_steps=special_svd_steps,   # ★ 前 5 个 step 用 special SVD
        log_interval = log_interval,  # cos similarity

        special_param_names=special_param_names if len(special_param_names) > 0 else None,  # <-- 传入

        # === 新增：两个开关 ===
        enable_svd_metric=bool(enable_svd_metric),
        enable_special_svd=bool(enable_special_svd),
        enable_cos_metric=bool(enable_cos_metric),

        special_svd_interval=special_svd_interval,

        
        # ---- 新增：阈值 ----
        special_cos_threshold=float(args.special_cos_threshold),

        )

    
    save_config(training_args, cfg, out_dir)


    # === 手动创建 lr_scheduler（简洁 & 保留 warmup/cosine） ===
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / training_args.per_device_train_batch_size / training_args.gradient_accumulation_steps)
    num_training_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = DualLossMuonTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        lambda_r=lambda_r,
        optimizers=(optimizer, lr_scheduler),
        # callbacks=[SaveAfterNEpochsCallback()],  # ← 加这一行
    )


    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Plot training loss
    steps = [log['step'] for log in trainer.state.log_history if 'loss' in log]
    losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    plt.figure(figsize=(8,5))
    plt.plot(steps, losses, label="Training Loss")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("Training Loss over Steps"); plt.legend(); plt.grid(True)
    fig_path = os.path.join(out_dir, "loss_curve_pretrain.png")
    plt.savefig(fig_path); plt.show()
    # 保存 loss 数值到 json
    loss_values = {
        "step": steps,
        "loss": losses,
        "l_f": trainer.loss_log["l_f"],
        "l_r": trainer.loss_log["l_r"]
    }
    loss_file = os.path.join(out_dir, "loss_values.json")
    with open(loss_file, "w") as f:
        json.dump(loss_values, f, indent=2)
    print(f"Loss values saved to {loss_file}")


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
                        # 采样（每10步）
                        "steps_sample": m.get("steps_sample", []),
                        "ip_M_sample": m.get("ip_M_sample", []),
                        "cos_M_sample": m.get("cos_M_sample", []),
                        "ip_g_sample": m.get("ip_g_sample", []),
                        "cos_g_sample": m.get("cos_g_sample", []),
                        # 全量（每步）
                        # "steps_all": m.get("steps_all", []),
                        # "cos_M_all": m.get("cos_M_all", []),
                        # "cos_g_all": m.get("cos_g_all", []),
                    }
    metrics_file = os.path.join(out_dir, "muon_dual_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print(f"Muon dual metrics saved to {metrics_file}")




# ======================
# CLI
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune ChatGLM with Muon (dual-loss momentum)")
    parser.add_argument("--model_dir", type=str,
                        default="/home/huang341/myspace/GLM-VOICE/glm-9b-voice-model",
                        help="路径到预训练模型目录")
    parser.add_argument("--data_file", type=str,
                        default="/home/huang341/myspace/GLM_unlearn/GLM-4-Voice-main/muon/data_process/cose/dataset/cose_train_processed_2.json",
                        help="训练数据文件路径")
    parser.add_argument("--out_dir", type=str,
                        default="/home/huang341/myspace/GLM_unlearn/GLM-4-Voice-main/muon/data_process/cose/sft_models/muon_v1",
                        help="输出保存目录")
    
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/home/huang341/myspace/GLM-VOICE/glm-9b-voice-model",
                        help="tokenizer 目录（默认和 model_dir 一致）")
    parser.add_argument("--batch_size", type=int, default=16, help="每卡 batch size")

    parser.add_argument("--epochs", type=int, default=8, help="训练 epoch 数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率（给调度器用）")
    parser.add_argument("--max_len", type=int, default=800, help="最大序列长度")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="是否从 checkpoint 恢复训练")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减（Trainer级别）")


    # parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    # 为了复现
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="梯度裁剪阈值")
    
    
    parser.add_argument("--lambda_r", type=float, default=2.0, help="Audio loss weight (当前实现未显式乘，但可在损失处使用)")

    # 新增：Muon 与 RMSNorm 的 WD、以及 Kimi scaling 的参数
    parser.add_argument("--muon_weight_decay", type=float, default=0.1,
                        help="Muon 侧权重衰减 (默认 0.1)")
    parser.add_argument("--rmsnorm_weight_decay", type=float, default=0.1,
                        help="RMSNorm gamma 权重衰减 (默认 0.1)")
    parser.add_argument("--muon_kimi_c", type=float, default=0.2,
                        help="Kimi 缩放系数 (默认 0.2; 建议扫 0.15–0.25)")
    parser.add_argument("--muon_use_kimi_scaling", type=int, default=1,
                        help="是否启用 Kimi 缩放：1 开启（默认），0 关闭")

                        
    parser.add_argument("--metric_interval", type=int, default=200,
                        help="sti度量写盘的步长间隔，步号能被该值整除时才写入（默认 10000）")
    parser.add_argument("--special_svd_steps", type=int, default=1,
                        help="启用“特殊 SVD 融合更新”的步数窗口（>=0 生效；-1 表示关闭，默认 -1）")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="优化器内部 cos  采样日志的步长间隔（默认 10）")

    parser.add_argument("--special_layers_file", type=str, default="/home/huang341/myspace/GLM_unlearn/GLM-4-Voice-main/muon/layer.txt",
                        help="包含待选择特殊层名的txt（每行一个完整参数名）")
    parser.add_argument("--special_layers_idx", type=str, default="[153]",
                        help="按1-based索引选择特殊层，如 \"[1]\"、\"[3 5]\"、\"1,3,5\"；空串表示不显式指定（回退正则）")

    parser.add_argument("--enable_special_svd", type=int, default=1,
                        help="是否启用特殊层的v1")
    parser.add_argument("--enable_cos_metric", type=int, default=0,
                        help="是否采样并记录 cos  ")
    parser.add_argument("--enable_svd_metric", type=int, default=0,
                        help="是否启用 sti")


    parser.add_argument("--special_svd_interval", type=int, default=1,
    help="特殊层 SVD 融合更新的触发步间隔（默认 1：每步触发；例如 10 表示每 10 步触发一次）")

    
    parser.add_argument(
        "--train_samples", type=int, default=-1,
        help="仅使用前 N 条样本进行训练；<=0 表示使用全部数据（默认 -1）"
    )

    parser.add_argument("--special_cos_threshold", type=float, default=1.0,
    help="cos(M_dec) 触发 SPECIAL-SVD 的阈值（默认 0.0；例如 -0.05、-0.1 更保守）")


    

    args = parser.parse_args()
    pretrain(
        model_dir=args.model_dir,
        data_file=args.data_file,
        out_dir=args.out_dir,
        tokenizer_dir=args.tokenizer_dir,
        per_device_batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        resume_from_checkpoint=args.resume_from_checkpoint,
        lambda_r=args.lambda_r,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        muon_weight_decay=args.muon_weight_decay,
        rmsnorm_weight_decay=args.rmsnorm_weight_decay,
        muon_kimi_c=args.muon_kimi_c,
        muon_use_kimi_scaling=args.muon_use_kimi_scaling,

        
        
        metric_interval=args.metric_interval,
        special_svd_steps=args.special_svd_steps,
        log_interval=args.log_interval,



        # ---- NEW: pass through ----
        special_layers_file=args.special_layers_file,
        special_layers_idx=args.special_layers_idx,

        
        enable_special_svd=args.enable_special_svd,
        enable_cos_metric=args.enable_cos_metric,
        enable_svd_metric=args.enable_svd_metric,

        special_svd_interval=args.special_svd_interval,

        train_samples=args.train_samples,



    )

 


 