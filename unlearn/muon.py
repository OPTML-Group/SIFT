# ===== Muon + LowRankMuon unified implementation ======================================
import re
import torch
import torch.distributed as dist

TARGET_PARAM_NAMES = [
    "model.layers.5.mlp.down_proj.weight",
    "model.layers.6.mlp.down_proj.weight",
    "model.layers.7.mlp.down_proj.weight",
]
TARGET_PARAM_NAME_SET = set(TARGET_PARAM_NAMES)


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

import json
import os
import time

class StatsWriter:
    """
    Rank0-only JSONL writer.
    Each call to write() appends ONE line (one step).
    """
    def __init__(self, path: str, flush_every: int = 1):
        self.path = path
        self.flush_every = flush_every
        self._fh = None
        self._counter = 0

        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _open(self):
        if self._fh is None:
            self._fh = open(self.path, "a", buffering=1)

    def write(self, record: dict):
        self._open()
        self._fh.write(json.dumps(record) + "\n")
        self._counter += 1
        if self.flush_every > 0 and self._counter % self.flush_every == 0:
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

import time

class _Timer:
    """CPU wall timer."""
    def __init__(self): self.t0 = None
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt_ms = (time.perf_counter() - self.t0) * 1000.0

class _CudaTimer:
    """CUDA event timer (ms). Safe fallback to CPU if not cuda."""
    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = (device.type == "cuda")
        self.start = None
        self.end = None
        self.dt_ms = 0.0

    def __enter__(self):
        if self.use_cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end   = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.use_cuda:
            self.end.record()
            torch.cuda.synchronize(self.device)
            self.dt_ms = float(self.start.elapsed_time(self.end))
        else:
            self.dt_ms = (time.perf_counter() - self._t0) * 1000.0


@torch.no_grad()
def _sti_from_two_tasks(
    Ur: torch.Tensor, Vr: torch.Tensor,
    Uf: torch.Tensor, Vf: torch.Tensor,
    Sr: torch.Tensor, Sf: torch.Tensor,
) -> torch.Tensor:
    """
    STI between two rank-k tasks: UrVr^T vs UfVf^T
    """
    # 拼成 [Ur, Uf], [Vr, Vf]
    Uhat = torch.cat([Ur, Uf], dim=1)   # (rows, 2k)
    Vhat = torch.cat([Vr, Vf], dim=1)   # (cols, 2k)
    sigma = torch.cat([Sr, Sf], dim=0)  # (2k,)

    return _sti_from_UVhat(
        Uhat.to(torch.float32),
        Vhat.to(torch.float32),
        sigma.to(torch.float32),
    )


# Projected error (Frobenius norm)
@torch.no_grad()
def proj_err_fro(M2d, P, eps=1e-12):
    """
    err_fro = ||M - P.T @ (P @ M)||_F / ||M||_F
    return: (numerator, denominator, err)
    """
    # proj(M) = P^T (P M)
    
    PM = P @ M2d
    proj = P.mT @ PM
    resid = M2d - proj
    num = torch.linalg.norm(resid, ord="fro")
    den = torch.linalg.norm(M2d, ord="fro").clamp_min(eps)
    return num, den, num / den


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
                  muon_eps: float = 1e-15, number: int = 0, return_stats: bool = False, lowrank_method: str = "svd",
                  lowrank_opt_steps: int = 5,
                  lowrank_opt_lr: float = 1e-2):
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
    
    method = state.get("lowrank_method", lowrank_method)
    
    if method is None:
        method = "svd"

    if 'lrmuon_step' not in state:
        state['lrmuon_step'] = 0
    
    # print(lowrank_method)
    # print(method)
    
    need_refresh = (
        ('lrmuon_P' not in state) or
        (state['lrmuon_step'] % int(refresh_T) == 0) or
        (state.get('lrmuon_shape') != (rows, cols)) or
        (state.get('lrmuon_k') != k_eff) or
        (state.get('lrmuon_method') != method)
    )
    
    # print("Check here.")
    
    if need_refresh:
        
        # print(f"[LR-Muon] Refreshing P for param #{number}: shape=({rows},{cols}), k={k_eff}, method={method}")
        P, proj_ms = _build_lowrank_P(
            M2d=M2d,
            k_eff=k_eff,
            method=method,
            state=state,
            opt_steps=int(state.get("lowrank_opt_steps", 5)),
            opt_lr=float(state.get("lowrank_opt_lr", 1e-2)),
        )
        state['lrmuon_P'] = P
        state['lrmuon_k'] = k_eff
        state['lrmuon_shape'] = (rows, cols)
        state['lrmuon_method'] = method
        state['lrmuon_last_proj_ms'] = float(proj_ms)
    else:
        state['lrmuon_last_proj_ms'] = 0.0

    state['lrmuon_step'] += 1
    P = state['lrmuon_P']
    
    # low-rank muon: project → NS → back-project
    X = P @ M2d
    # Y = zeropower_via_newtonschulz5(X, steps=ns_steps, eps=muon_eps).to(P.dtype)
    
    if state.get("_timing_enabled", False):
        with _CudaTimer(X.device) as tt:
            Y = zeropower_via_newtonschulz5(X, steps=ns_steps, eps=muon_eps).to(P.dtype)
        state["lrmuon_last_ns_ms"] = float(tt.dt_ms)
    else:
        Y = zeropower_via_newtonschulz5(X, steps=ns_steps, eps=muon_eps).to(P.dtype)
        state["lrmuon_last_ns_ms"] = 0.0
    
    
    
    stat = None
    if return_stats:
        stat = {
            "P_refreshed": float(need_refresh),
            "M_rows": int(rows),
            "M_cols": int(cols),
            "P_k": int(P.size(0)),
            "P_rows": int(P.size(1)),
        }

        # (A) projection residual on M  —— 只算一次
        num, den, err = proj_err_fro(M2d, P)
        stat.update({
            "err_num": num,
            "err_den": den,
            "err": err,
        })

        # (B) end-to-end sign operator error —— 复用 Y=msign(PM)，只额外算一次 msign(M)
        # 注意：这里用的是“未缩放的 Y”，因为缩放属于 update，不属于 msign 本体
        S_full = zeropower_via_newtonschulz5(M2d, steps=ns_steps, eps=muon_eps)

        S_proj = P.mT @ Y  # P^T msign(PM)
        resid_s = S_full - S_proj
        s_num = torch.linalg.norm(resid_s, ord="fro")
        s_den = torch.linalg.norm(S_full, ord="fro").clamp_min(1e-12)
        s_err = s_num / s_den

        stat.update({
            "sign_err_num": s_num,
            "sign_err_den": s_den,
            "sign_err": s_err,
        })
    
    # optional scale in subspace
    scale = _muon_scale(k_eff, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    Y = Y * scale

    update2d = P.mT @ Y
    update = update2d.view_as(M)
    param.add_(update, alpha=-lr)
    
    if return_stats:
        return stat
    return None


def _muon_scale(rows, cols, use_kimi_scaling: bool = False, kimi_c: float = 0.2):
    if use_kimi_scaling:
        return kimi_c * (max(rows, cols) ** 0.5)
    else:
        return (max(1.0, rows / max(cols, 1))) ** 0.5  # sqrt(Out/In)


@torch.no_grad()
def _orth_rows_from_cols(Q_cols: torch.Tensor) -> torch.Tensor:
    """
    Q_cols: (rows, k) with orthonormal columns
    return: P = Q^T of shape (k, rows) with orthonormal rows
    """
    return Q_cols.mT.contiguous()

@torch.no_grad()
def _build_lowrank_P(M2d: torch.Tensor, k_eff: int, method: str,
                     state: dict,
                     opt_steps: int = 5,
                     opt_lr: float = 1e-2) -> torch.Tensor:
    """
    Returns P with shape (k, rows), row-orthonormal.
    M2d: (rows, cols)
    """
    rows, cols = M2d.shape
    method = (method or "svd").lower()

    dev = M2d.device
    
    P = None
    with _CudaTimer(dev) as tt:
    
        # --- (1) svd: exact top-k left singular vectors ---
        if method == "svd":
            print("svd method called.")
            U, S, Vh = torch.linalg.svd(M2d.to(torch.float32), full_matrices=False)
            Q = U[:, :k_eff].contiguous().to(device=M2d.device, dtype=M2d.dtype)  # (rows, k)
            P = _orth_rows_from_cols(Q)
            # return _orth_rows_from_cols(Q), tt.dt_ms

        # --- (2) random: random orthonormal subspace in R^{rows} ---
        if method == "random":
            A = torch.randn(rows, k_eff, device=M2d.device, dtype=torch.float32)
            Q, _ = torch.linalg.qr(A, mode="reduced")  # (rows, k)
            Q = Q.to(dtype=M2d.dtype)
            P = _orth_rows_from_cols(Q)
            # return _orth_rows_from_cols(Q), tt.dt_ms

        # --- (3) gs: randomized range finder (Gaussian sketch) approximating top-k subspace ---
        # Standard: Y = M @ Omega, Omega ~ N(0,1), then QR(Y) gives approx left subspace
        if method == "gs":
            print("gs method called.")
            Omega = torch.randn(cols, k_eff, device=M2d.device, dtype=torch.float32)
            Y = (M2d.to(torch.float32) @ Omega)  # (rows, k)
            Q, _ = torch.linalg.qr(Y, mode="reduced")
            Q = Q.to(device=M2d.device, dtype=M2d.dtype)
            P = _orth_rows_from_cols(Q)
            # return _orth_rows_from_cols(Q), tt.dt_ms

        # --- (4) optimize: placeholder hook (keep framework-ready) ---
        # You can later implement a projected-gradient / manifold optimization here.
        if method == "optimize":
            print("optimize method called.")

            device = M2d.device
            dtype_out = M2d.dtype

            # treat M as constant; normalize for stability
            M32 = M2d.detach().to(torch.float32)
            M32 = M32 / (M32.norm() + 1e-8)

            # init Q: random orthonormal columns
            Q0 = torch.randn(rows, k_eff, device=device, dtype=torch.float32)
            Q, _ = torch.linalg.qr(Q0, mode="reduced")  # (rows,k)
            Q = Q.detach().requires_grad_(True)         # <-- IMPORTANT: leaf

            I = torch.eye(k_eff, device=device, dtype=torch.float32)

            opt_lambda = 0.1
            opt_reorth_every = int(state.get("lowrank_opt_reorth_every", 1))  # optional

            # CRITICAL: lrmuon_update is @torch.no_grad(); must re-enable autograd here
            with torch.enable_grad():
                for t in range(int(opt_steps)):
                    # QQ^T M = Q (Q^T M)
                    QtM = Q.mT @ M32
                    projM = Q @ QtM

                    recon_loss = (projM - M32).pow(2).sum()
                    ortho_loss = (Q.mT @ Q - I).pow(2).sum()
                    loss = recon_loss + float(opt_lambda) * ortho_loss

                    loss.backward()

                    with torch.no_grad():
                        # GD step
                        Q -= float(opt_lr) * Q.grad
                        Q.grad.zero_()

                        # re-orthonormalize (retraction)
                        if opt_reorth_every > 0 and ((t + 1) % opt_reorth_every == 0):
                            Q, _ = torch.linalg.qr(Q, mode="reduced")
                            Q = Q.detach().requires_grad_(True)  # <-- IMPORTANT: leaf again

            Q_final = Q.detach().to(dtype_out)
            P = Q_final.mT.contiguous()   # (k, rows)
            # return P, tt.dt_ms

    proj_ms = float(tt.dt_ms)
    return P, proj_ms
    # raise ValueError(f"Unknown lowrank_method: {method}")


# @torch.compile
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
def _uvhat_from_UV(U: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert full-size U:(rows,k), V:(cols,k) into Uhat:(rows,2k), Vhat:(cols,2k)
    by stacking [U, U] and [V, V]. (v0-topk uses sum of two rank-k terms; Uhat/Vhat is just a
    convenient "combined" representation for logging.)
    """
    Uhat = torch.cat([U, U], dim=1)   # (rows, 2k)
    Vhat = torch.cat([V, V], dim=1)   # (cols, 2k)
    return Uhat, Vhat


@torch.no_grad()
def svd_subspace_topk_with_s(M2d: torch.Tensor, k: int):
    rows, cols = M2d.shape
    r = int(min(rows, cols))
    k_eff = int(max(1, min(int(k), r)))

    M32 = M2d.to(torch.float32)
    U, S, Vh = torch.linalg.svd(M32, full_matrices=False)
    V = Vh.transpose(-2, -1)
    return (
        U[:, :k_eff].contiguous(),     # (rows, k)
        S[:k_eff].contiguous(),        # (k,)
        V[:, :k_eff].contiguous(),     # (cols, k)
    )


@torch.no_grad()
def _sti_from_UVhat(U_hat: torch.Tensor, V_hat: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    STI = || (U^T U - I) Σ (V^T V - I) ||_1 / [k_tot (k_tot-1)]
    elementwise L1 sum.
    """
    
    U = U_hat.to(torch.float32)
    V = V_hat.to(torch.float32)
    s = sigma.to(torch.float32)

    k_tot = int(s.numel())
    if k_tot <= 1:
        return torch.zeros((), dtype=torch.float32, device=U.device)

    I = torch.eye(k_tot, dtype=torch.float32, device=U.device)
    UU_minus_I = (U.mT @ U) - I          # (k,k)
    VV_minus_I = (V.mT @ V) - I          # (k,k)

    SV = s.unsqueeze(1) * VV_minus_I
    M = UU_minus_I @ SV
    sti_raw = M.abs().sum()
    denom = k_tot * (k_tot - 1)
    return (sti_raw / (denom + 1e-12)).to(torch.float32)


@torch.no_grad()
def muon_update(grad, momentum, *,
                beta=0.95, ns_steps=5, nesterov=True,
                use_kimi_scaling: bool = False, kimi_c: float = 0.2, timing: bool = False):
    if grad is None:
        return None, 0.0

    # 1) momentum
    momentum.mul_(beta).add_(grad)
    M = grad.add(momentum, alpha=beta) if nesterov else momentum

    # 2) 2D
    M2d = _flat2d(M)
    rows, cols = M2d.shape

    if timing:
        with _CudaTimer(M2d.device) as tt:
            U = zeropower_via_newtonschulz5(M2d, steps=ns_steps)
        ns_ms = tt.dt_ms
    else:
        U = zeropower_via_newtonschulz5(M2d, steps=ns_steps)
        ns_ms = 0.0
    
    # 3) orthogonalize
    # U = zeropower_via_newtonschulz5(M2d, steps=ns_steps)

    # 4) scale
    scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    update2d = U * scale
    return update2d.view_as(M), ns_ms


@torch.no_grad()
def msign_via_svd(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Exact polar factor via SVD:
      A = P diag(s) Q^T  =>  msign(A) := P Q^T
    Works for any 2D A (including tall/skinny).
    """
    # do SVD in fp32 for stability
    A32 = A.to(torch.float32)
    # full_matrices=False gives P:(m,k), Q:(n,k) if m>=n, etc. Still OK.
    P, S, Qh = torch.linalg.svd(A32, full_matrices=False)
    Q = Qh.transpose(-2, -1)
    # P @ Q^T has shape (m,n) only if P:(m,r), Q:(n,r) and we do P @ Q^T -> (m,n)
    out32 = P @ Q.transpose(-2, -1)
    return out32.to(dtype=A.dtype, device=A.device)


@torch.no_grad()
def svd_subspace(M2d: torch.Tensor):
    """
    Return (U, V) from SVD of M2d:
      M = U Σ V^T
    We only need U and V (no Σ).
    """
    M32 = M2d.to(torch.float32)
    U, S, Vh = torch.linalg.svd(M32, full_matrices=False)
    V = Vh.transpose(-2, -1)
    return U, V  # fp32, shapes: U:(rows,r), V:(cols,r)


def _fmt_head(x: torch.Tensor, n: int = 8) -> str:
        # flatten -> take first n -> format like 1.00e+03 / 1.34e-0
    h = x.detach().float().reshape(-1)[:n].cpu().numpy()
    return " ".join([f"{v: .2e}" for v in h])  # 1000 -> 1.00e+03, 0.00134 -> 1.34e-03

def _fmt_shape(x: torch.Tensor) -> str:
    return f"shape={tuple(x.shape)} ndim={x.ndim} dtype={x.dtype} device={x.device}"



@torch.no_grad()
def muon_update_v1_split(
    gf: torch.Tensor,
    gr: torch.Tensor,
    buf_f: torch.Tensor,
    buf_r: torch.Tensor,
    *,
    beta: float,
    nesterov: bool,
    weight_decay: float,
    lr: float,
    use_kimi_scaling: bool = False,
    kimi_c: float = 0.2,
):
    """
    v1 Eq.(20) for ONE param tensor:
    - Update momentum buffers separately for forget/retain
    - Build task matrices Mf, Mr (Nesterov if enabled)
    - SVD => (Uf,Vf), (Ur,Vr)
    - Concatenate: Uhat=[Ur,Uf], Vhat=[Vr,Vf]
    - Orthogonalize: Uperp=msign(Uhat), Vperp=msign(Vhat)
    - Final update2d = Uperp @ Vperp^T (same shape as M2d)
    - Apply Muon scaling, then return update in original shape
    """
    if gf is None and gr is None:
        return None
    
    # print("forget grad", gf)
    # print("retain grad", gr)
    

    # ... inside your loop where you have gr ...
    if gr is not None:
        print("[retain grad]", _fmt_shape(gr), "| head:", _fmt_head(gr, n=10))
    else:
        print("[retain grad] None")
    
    # import sys
    # sys.exit()

    # -------- 1) momentum buffers update (separate) --------
    if gf is not None:
        buf_f.mul_(beta).add_(gf)
    else:
        # keep buf_f as-is (or you can decay only; current keeps)
        pass

    if gr is not None:
        buf_r.mul_(beta).add_(gr)
    else:
        pass

    # -------- 2) build M_f, M_r (nesterov uses grad + beta*buf) --------
    # IMPORTANT: match your muon_update definition:
    # momentum.mul_(beta).add_(grad)
    # M = grad + beta*momentum  (if nesterov)
    Mf = None
    Mr = None
    if gf is not None:
        Mf = gf.add(buf_f, alpha=beta) if nesterov else buf_f
    if gr is not None:
        Mr = gr.add(buf_r, alpha=beta) if nesterov else buf_r

    
    # print("forget Mf", Mf)
    # print("retain Mg", Mr)
    
    # if one side missing, fallback to that side's vanilla muon update direction
    # (still using v1 machinery is possible, but needs two tasks to merge)
    if Mf is None:
        M2d = _flat2d(Mr)
        U = msign_via_svd(M2d)  # exact polar factor of M
        rows, cols = M2d.shape
        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
        update2d = U * scale
        return update2d.view_as(Mr)

    if Mr is None:
        M2d = _flat2d(Mf)
        U = msign_via_svd(M2d)
        rows, cols = M2d.shape
        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
        update2d = U * scale
        return update2d.view_as(Mf)

    # -------- 3) SVD subspaces --------
    M2d_f = _flat2d(Mf)
    M2d_r = _flat2d(Mr)
    rows, cols = M2d_r.shape
    assert M2d_f.shape == (rows, cols), "forget/retain M shape mismatch"

    # print("M2d_f", M2d_f.shape)
    
    Ur, Vr = svd_subspace(M2d_r)  # fp32
    Uf, Vf = svd_subspace(M2d_f)  # fp32

    # -------- 4) concatenate then orthogonalize (Eq.19 / Eq.20) --------
    # Uhat: (rows, r_r + r_f), Vhat: (cols, r_r + r_f)
    Uhat = torch.cat([Ur, Uf], dim=1).to(device=M2d_r.device, dtype=M2d_r.dtype)
    Vhat = torch.cat([Vr, Vf], dim=1).to(device=M2d_r.device, dtype=M2d_r.dtype)
    
    # print("Uhat", Uhat.shape)

    Uperp = msign_via_svd(Uhat)  # (rows, rows)?? careful:
    Vperp = msign_via_svd(Vhat)
    
    # print("Uperp", Uperp.shape)

    # NOTE:
    # msign_via_svd(Uhat) returns shape (rows, rows) if Uhat is (rows, k) and SVD gives P:(rows,rows)?? No:
    # with full_matrices=False, P:(rows, k), Q:(k, k), so P@Q^T -> (rows, k). That's what we want.
    # So Uperp: (rows, k), Vperp: (cols, k), and Uperp@Vperp^T -> (rows, cols).

    update2d = Uperp @ Vperp.transpose(-2, -1)  # (rows, cols)
    
    # print("update2d", update2d.shape)

    # -------- 5) scale (Muon scaling) --------
    scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    update2d = update2d * scale

    # import sys
    # sys.exit()
    return update2d.view_as(Mr)  # same shape as grad/momentum tensor


@torch.no_grad()
def svd_subspace_topk(M2d: torch.Tensor, k: int):
    """
    Thin SVD subspace, truncated to top-k singular vectors.

    Input:
      M2d: (rows, cols)
      k:   desired rank

    Return:
      U_k: (rows, k_eff)  fp32
      V_k: (cols, k_eff)  fp32
    """
    rows, cols = M2d.shape
    r = int(min(rows, cols))
    k_eff = int(max(1, min(int(k), r)))

    M32 = M2d.to(torch.float32)
    U, S, Vh = torch.linalg.svd(M32, full_matrices=False)  # U:(rows,r), Vh:(r,cols)
    V = Vh.transpose(-2, -1)                                # V:(cols,r)

    return U[:, :k_eff].contiguous(), V[:, :k_eff].contiguous()


# ====================== STI (YOUR REQUIRED PIPELINE) ======================
# External: assemble task matrices (UrVr^T, UfVf^T)
# Internal: SVD(task_matrix) -> (U,S,V) then compute STI

@torch.no_grad()
def _sti_from_two_task_mats_strict(
    Mr_task2d: torch.Tensor,   # (rows, cols)
    Mf_task2d: torch.Tensor,   # (rows, cols)
    k: int | None = None,
) -> torch.Tensor:
    assert Mr_task2d.ndim == 2 and Mf_task2d.ndim == 2
    assert Mr_task2d.shape == Mf_task2d.shape

    rows, cols = Mr_task2d.shape
    r = int(min(rows, cols))
    k_eff = r if (k is None) else int(max(1, min(int(k), r)))

    # SVD inside STI (fp32)
    Ur, Sr, Vhr = torch.linalg.svd(Mr_task2d.to(torch.float32), full_matrices=False)
    Vr = Vhr.transpose(-2, -1)

    Uf, Sf, Vhf = torch.linalg.svd(Mf_task2d.to(torch.float32), full_matrices=False)
    Vf = Vhf.transpose(-2, -1)

    # top-k
    Ur = Ur[:, :k_eff].contiguous()
    Vr = Vr[:, :k_eff].contiguous()
    Sr = Sr[:k_eff].contiguous()

    Uf = Uf[:, :k_eff].contiguous()
    Vf = Vf[:, :k_eff].contiguous()
    Sf = Sf[:k_eff].contiguous()

    Uhat = torch.cat([Ur, Uf], dim=1)          # (rows, 2k)
    Vhat = torch.cat([Vr, Vf], dim=1)          # (cols, 2k)
    sigma = torch.cat([Sr, Sf], dim=0)         # (2k,)

    return _sti_from_UVhat(Uhat, Vhat, sigma)


@torch.no_grad()
def muon_update_v1_split_topk(
    gf: torch.Tensor,
    gr: torch.Tensor,
    buf_f: torch.Tensor,
    buf_r: torch.Tensor,
    *,
    beta: float,
    nesterov: bool,
    weight_decay: float,
    lr: float,
    v1_k: int = 256,                 # NEW: top-k truncation
    use_kimi_scaling: bool = False,
    kimi_c: float = 0.2,
    # Optional: if True, enforce 2*v1_k <= min(rows, cols) by shrinking k.
    enforce_total_rank_cap: bool = True,
):

    if gf is None and gr is None:
        return None

    # -------- 1) momentum buffers update (separate) --------
    if gf is not None:
        buf_f.mul_(beta).add_(gf)
    if gr is not None:
        buf_r.mul_(beta).add_(gr)

    # -------- 2) build M_f, M_r (nesterov uses grad + beta*buf) --------
    Mf = (gf.add(buf_f, alpha=beta) if nesterov else buf_f) if gf is not None else None
    Mr = (gr.add(buf_r, alpha=beta) if nesterov else buf_r) if gr is not None else None

    # fallback if one side missing: vanilla Muon direction on the available side
    if Mf is None and Mr is not None:
        M2d = _flat2d(Mr)
        U = msign_via_svd(M2d)
        rows, cols = M2d.shape
        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
        return (U * scale).view_as(Mr)

    if Mr is None and Mf is not None:
        M2d = _flat2d(Mf)
        U = msign_via_svd(M2d)
        rows, cols = M2d.shape
        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
        return (U * scale).view_as(Mf)

    # -------- 3) SVD subspaces (TOP-K) --------
    M2d_f = _flat2d(Mf)
    M2d_r = _flat2d(Mr)
    rows, cols = M2d_r.shape
    assert M2d_f.shape == (rows, cols), "forget/retain M shape mismatch"

    r = int(min(rows, cols))
    k_eff = int(max(1, min(int(v1_k), r)))

    # IMPORTANT: Uhat has width (k_r + k_f) = 2*k_eff.
    # For stability & cost, optionally cap so 2*k_eff <= r (or <= rows/cols).
    if enforce_total_rank_cap:
        k_eff = int(max(1, min(k_eff, r // 2)))

    print("k_eff", k_eff)
    
    # import sys
    # sys.exit()
    
    # Ur, Vr = svd_subspace_topk(M2d_r, k=k_eff)  # fp32: Ur:(rows,k), Vr:(cols,k)
    # Uf, Vf = svd_subspace_topk(M2d_f, k=k_eff)  # fp32: Uf:(rows,k), Vf:(cols,k)
    
    Ur, Sr, Vr = svd_subspace_topk_with_s(M2d_r, k=k_eff)
    Uf, Sf, Vf = svd_subspace_topk_with_s(M2d_f, k=k_eff)

    # -------- 4) concatenate then orthogonalize (Eq.19 / Eq.20) --------
    Uhat = torch.cat([Ur, Uf], dim=1).to(device=M2d_r.device, dtype=M2d_r.dtype)  # (rows, 2k)
    Vhat = torch.cat([Vr, Vf], dim=1).to(device=M2d_r.device, dtype=M2d_r.dtype)  # (cols, 2k)

    Uperp = msign_via_svd(Uhat)  # (rows, 2k)
    Vperp = msign_via_svd(Vhat)  # (cols, 2k)
    
    # # -------- STRICT STI(before): assemble UrVr^T & UfVf^T then SVD inside STI --------
    # Mr_task2d_before = (Ur @ Vr.transpose(-2, -1)).to(device=M2d_r.device, dtype=M2d_r.dtype)
    # Mf_task2d_before = (Uf @ Vf.transpose(-2, -1)).to(device=M2d_r.device, dtype=M2d_r.dtype)
    # sti_before = _sti_from_two_task_mats_strict(Mr_task2d_before, Mf_task2d_before, k=k_eff)

    # -------- STRICT STI(after): split Uperp/Vperp into two "tasks" (first k, last k) --------
    # Uperp_r = Uperp[:, :k_eff]
    # Vperp_r = Vperp[:, :k_eff]
    # Uperp_f = Uperp[:, k_eff: 2 * k_eff]
    # Vperp_f = Vperp[:, k_eff: 2 * k_eff]

    # Mr_task2d_after = (Uperp_r @ Vperp_r.transpose(-2, -1)).to(device=M2d_r.device, dtype=M2d_r.dtype)
    # Mf_task2d_after = (Uperp_f @ Vperp_f.transpose(-2, -1)).to(device=M2d_r.device, dtype=M2d_r.dtype)
    # sti_after = _sti_from_two_task_mats_strict(Mr_task2d_after, Mf_task2d_after, k=k_eff)
    
    update2d = Uperp @ Vperp.transpose(-2, -1)  # (rows, cols)

    # -------- 5) scale (Muon scaling) --------
    use_kimi_scaling = False
    scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    update2d = update2d * scale

    # return update2d.view_as(Mr)
    return update2d.view_as(Mr)
# , sti_before, sti_after


@torch.no_grad()
def muon_update_v2_split_leftU(
    gf: torch.Tensor,
    gr: torch.Tensor,
    buf_c: torch.Tensor,
    buf_f: torch.Tensor,
    buf_r: torch.Tensor,
    *,
    beta: float,
    nesterov: bool,
    lr: float,
    weight_decay: float,
    v2_common_k: int,
    use_kimi_scaling: bool = False,
    kimi_c: float = 0.2,
):
    """
    V2 (left-projection-only) for ONE param tensor.

    0) Maintain 3 buffers:
       - buf_c: common buffer for Mc (single buffer over g_c = gf + gr)
       - buf_f: (optional) forget buffer for Mf (kept for symmetry / fallback)
       - buf_r: (optional) retain buffer for Mr (kept for symmetry / fallback)

    1) Build Mf, Mr:
       buf_f <- beta*buf_f + gf (if gf exists, else decay only)
       buf_r <- beta*buf_r + gr (if gr exists, else decay only)
       Mf = gf + beta*buf_f (if nesterov) else buf_f
       Mr = gr + beta*buf_r (if nesterov) else buf_r

    2) Build Mc via SINGLE buffer:
       g_c = (gf if not None else 0) + (gr if not None else 0)
       buf_c <- beta*buf_c + g_c   (ALWAYS decay, even if g_c=0)
       Mc = g_c + beta*buf_c (if nesterov) else buf_c

    3) SVD(Mc2d) -> (Uc,Vc) then take top-k = v2_common_k
    4) Left-project task matrices:
       Mf_hat = (I - Uc Uc^T) Mf
       Mr_hat = (I - Uc Uc^T) Mr
       (only left projection, as you requested)

    5) SVD(Mr_hat), SVD(Mf_hat) to get (Ur_hat,Vr_hat), (Uf_hat,Vf_hat)
    6) Concatenate:
       Uhat = [Uc, Ur_hat, Uf_hat]
       Vhat = [Vc, Vr_hat, Vf_hat]
       Uperp = msign(Uhat), Vperp = msign(Vhat)
       update2d = Uperp @ Vperp^T
       scale + reshape back
    """

    if gf is None and gr is None:
        # still decay common buffer to keep "time axis" consistent
        buf_c.mul_(beta)
        buf_f.mul_(beta)
        buf_r.mul_(beta)
        return None

    # ---------- (1) update buf_f / buf_r (decay even if missing) ----------
    buf_f.mul_(beta)
    buf_r.mul_(beta)
    if gf is not None:
        buf_f.add_(gf)
    if gr is not None:
        buf_r.add_(gr)

    Mf = (gf.add(buf_f, alpha=beta) if nesterov else buf_f) if gf is not None else None
    Mr = (gr.add(buf_r, alpha=beta) if nesterov else buf_r) if gr is not None else None

    # fallback if one side missing (keep behavior consistent with your v1 fallbacks)
    if Mf is None and Mr is not None:
        M2d = _flat2d(Mr)
        U = msign_via_svd(M2d)
        rows, cols = M2d.shape
        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
        return (U * scale).view_as(Mr)

    if Mr is None and Mf is not None:
        M2d = _flat2d(Mf)
        U = msign_via_svd(M2d)
        rows, cols = M2d.shape
        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
        return (U * scale).view_as(Mf)

    # ---------- (2) build Mc via SINGLE buffer buf_c ----------
    # g_c = gf + gr (missing treated as 0)
    g_c = None
    if gf is None:
        g_c = gr
    elif gr is None:
        g_c = gf
    else:
        g_c = gf + gr

    # ALWAYS decay common buffer; add g_c if exists
    buf_c.mul_(beta)
    if g_c is not None:
        buf_c.add_(g_c)

    Mc = g_c.add(buf_c, alpha=beta) if (nesterov and g_c is not None) else buf_c

    # ---------- (3) common subspace from SVD(Mc) ----------
    Mc2d = _flat2d(Mc)
    rows, cols = Mc2d.shape

    # k for common subspace (cap safely)
    k = int(max(1, min(int(v2_common_k), rows, cols)))

    Uc_full, Vc_full = svd_subspace(Mc2d)  # fp32
    Uc = Uc_full[:, :k].to(device=Mc2d.device, dtype=Mc2d.dtype).contiguous()  # (rows,k)
    Vc = Vc_full[:, :k].to(device=Mc2d.device, dtype=Mc2d.dtype).contiguous()  # (cols,k)

    # ---------- (4) LEFT projection only ----------
    # (I - UcUc^T) M  implemented as  M - Uc (Uc^T M)
    Mf2d = _flat2d(Mf)
    Mr2d = _flat2d(Mr)
    assert Mf2d.shape == Mr2d.shape == (rows, cols), "Mf/Mr/Mc shape mismatch"

    # project: M_hat = M - Uc (Uc^T M)
    Mf_hat = Mf2d - Uc @ (Uc.mT @ Mf2d)
    Mr_hat = Mr2d - Uc @ (Uc.mT @ Mr2d)

    # ---------- (5) SVD task-specific pieces ----------
    Ur_hat, Vr_hat = svd_subspace(Mr_hat)
    Uf_hat, Vf_hat = svd_subspace(Mf_hat)

    Ur_hat = Ur_hat.to(device=Mc2d.device, dtype=Mc2d.dtype)
    Vr_hat = Vr_hat.to(device=Mc2d.device, dtype=Mc2d.dtype)
    Uf_hat = Uf_hat.to(device=Mc2d.device, dtype=Mc2d.dtype)
    Vf_hat = Vf_hat.to(device=Mc2d.device, dtype=Mc2d.dtype)

    # ---------- (6) concatenate + msign ----------
    Uhat = torch.cat([Uc, Ur_hat, Uf_hat], dim=1)  # (rows, k + rr + rf)
    Vhat = torch.cat([Vc, Vr_hat, Vf_hat], dim=1)  # (cols, k + rr + rf)

    Uperp = msign_via_svd(Uhat)
    Vperp = msign_via_svd(Vhat)

    update2d = Uperp @ Vperp.transpose(-2, -1)  # (rows, cols)

    scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    update2d = update2d * scale
    return update2d.view_as(Mr)


@torch.no_grad()
def muon_update_v0_svd_concat(
    gf: torch.Tensor,
    gr: torch.Tensor,
    buf_f: torch.Tensor,
    buf_r: torch.Tensor,
    *,
    beta: float,
    nesterov: bool,
    lr: float,
    weight_decay: float,
    v0_k: int = 256,                 # NEW
    use_kimi_scaling: bool = False,
    kimi_c: float = 0.2,
    enforce_total_rank_cap: bool = True,  # optional safety
    return_uvhat: bool = False,   # NEW
):
    """
    True V0:
    update2d = Ur_k Vr_k^T + Uf_k Vf_k^T
    where (Ur, Vr) from SVD(Mr2d), (Uf, Vf) from SVD(Mf2d),
    and we only take the first k vectors (top-k).

    Notes:
    - No msign call; U,V come from SVD directly.
    - We ignore Sigma (same as msign definition U V^T).
    - Scale is applied ONCE to the final sum (like "msign(Mr)+msign(Mf)" then scaling).
    """

    use_kimi_scaling = True
    
    if gf is None and gr is None:
        return None

    # -------- 1) momentum buffers update (separate) --------
    if gf is not None:
        buf_f.mul_(beta).add_(gf)
    if gr is not None:
        buf_r.mul_(beta).add_(gr)

    # -------- 2) build M_f, M_r (Nesterov consistent with your Muon code) --------
    Mf = (gf.add(buf_f, alpha=beta) if nesterov else buf_f) if gf is not None else None
    Mr = (gr.add(buf_r, alpha=beta) if nesterov else buf_r) if gr is not None else None

    # fallback: if only one side exists, reduce to that side's "U V^T"
    if Mf is None and Mr is not None:
        M2d = _flat2d(Mr)
        rows, cols = M2d.shape
        r = int(min(rows, cols))
        k_eff = int(max(1, min(int(v0_k), r)))
        U, V = svd_subspace_topk(M2d, k=k_eff)  # fp32
        U = U.to(device=M2d.device, dtype=M2d.dtype)
        V = V.to(device=M2d.device, dtype=M2d.dtype)
        update2d = U @ V.transpose(-2, -1)
        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
        return (update2d * scale).view_as(Mr)

    if Mr is None and Mf is not None:
        M2d = _flat2d(Mf)
        rows, cols = M2d.shape
        r = int(min(rows, cols))
        k_eff = int(max(1, min(int(v0_k), r)))
        U, V = svd_subspace_topk(M2d, k=k_eff)  # fp32
        U = U.to(device=M2d.device, dtype=M2d.dtype)
        V = V.to(device=M2d.device, dtype=M2d.dtype)
        update2d = U @ V.transpose(-2, -1)
        scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
        return (update2d * scale).view_as(Mf)

    # -------- 3) both sides exist: compute true V0 by SVD concat (UrVr^T + UfVf^T) --------
    M2d_f = _flat2d(Mf)
    M2d_r = _flat2d(Mr)
    rows, cols = M2d_r.shape
    assert M2d_f.shape == (rows, cols), "forget/retain M shape mismatch"

    r = int(min(rows, cols))
    k_eff = int(max(1, min(int(v0_k), r)))

    # optional safety: if you later extend to [Ur,Uf] width=2k logic, keep it stable
    # For current v0 (sum of two rank-k terms), no strict need; still can cap to r.
    if enforce_total_rank_cap:
        k_eff = int(max(1, min(k_eff, r)))  # trivial cap; keep hook for symmetry
    
    print("k_eff:", k_eff)
    
    # Ur,Sr,Vr ; Uf,Sf,Vf (你可以用 svd_subspace_topk_with_s)
    Ur, Sr, Vr = svd_subspace_topk_with_s(M2d_r, k=k_eff)
    Uf, Sf, Vf = svd_subspace_topk_with_s(M2d_f, k=k_eff)

    Ur = Ur.to(device=M2d_r.device, dtype=M2d_r.dtype)
    Vr = Vr.to(device=M2d_r.device, dtype=M2d_r.dtype)
    Uf = Uf.to(device=M2d_r.device, dtype=M2d_r.dtype)
    Vf = Vf.to(device=M2d_r.device, dtype=M2d_r.dtype)

    # v0 update (unchanged)
    update2d = (Ur @ Vr.transpose(-2, -1)) + (Uf @ Vf.transpose(-2, -1))
    scale = _muon_scale(rows, cols, use_kimi_scaling=use_kimi_scaling, kimi_c=kimi_c)
    update2d = update2d * scale
    update = update2d.view_as(Mr)

    ######### 
    Mr_task2d = (Ur @ Vr.transpose(-2, -1)).to(device=M2d_r.device, dtype=M2d_r.dtype)
    Mf_task2d = (Uf @ Vf.transpose(-2, -1)).to(device=M2d_r.device, dtype=M2d_r.dtype)

    sti_before = _sti_from_two_task_mats_strict(Mr_task2d, Mf_task2d, k=k_eff)

    return update, sti_before


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups, stats_path: str = None,
                 timing_path: str = None,
                 enable_muon_svd_logging: bool = False,
                 enable_lrmuon_metrics: bool = False,
                 timing_check_T: int = 1):
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
        
        self.stats_history = {}   # {global_step: {...}}
        self._cur_step = None
        self.stats_writer = None
        self.muon_stats_writer = None
        
        self.enable_muon_svd_logging = bool(enable_muon_svd_logging)
        self.enable_lrmuon_metrics   = bool(enable_lrmuon_metrics)
        self.enable_timing           = (timing_path is not None)
        self.timing_check_T          = int(max(1, timing_check_T))
        
        self.timing_writer = None
        if timing_path is not None and _dist_rank() == 0:
            self.timing_writer = StatsWriter(timing_path)
        
        if getattr(param_groups[0].get("configs", None), "M_stats_path", None):
            pass  
        
        if stats_path is not None and _dist_rank() == 0:
            self.stats_writer = StatsWriter(stats_path)
            
        muon_path = None
        for g in self.param_groups:
            if g.get("use_muon", False):
                muon_path = g.get("M_stats_path", "")
                break
        if muon_path and _dist_rank() == 0:
            self.muon_stats_writer = StatsWriter(muon_path)
            
        self.sti_writer = None
        sti_path = None
        for g in self.param_groups:
            if g.get("use_muon", False):
                sti_path = g.get("STI_path", "")
                break
        if sti_path and _dist_rank() == 0:
            self.sti_writer = StatsWriter(sti_path, flush_every=1)  # 每条都 flush，最“即时”
            
    def set_step(self, step: int):
        self._cur_step = int(step)
        self._muon_ns_ms = 0.0
        self._lrmuon_ns_ms = 0.0
        self._lrmuon_proj_ms = {}   # method -> ms
        self._step_wall_ms = None
        self._step_breakdown = {}

    def write_timing_record(self):
        if (not self.enable_timing) or (self.timing_writer is None) or (self._cur_step is None):
            return
        if int(self._cur_step) % int(self.timing_check_T) != 0:
            return
        if _dist_rank() != 0:
            return

        rec = {
            "step": int(self._cur_step),
            "time": time.time(),
            "step_wall_ms": (None if self._step_wall_ms is None else float(self._step_wall_ms)),
            "muon_ns_ms_sum": float(self._muon_ns_ms),
            "lrmuon_ns_ms_sum": float(self._lrmuon_ns_ms),
            "lrmuon_proj_ms_by_method_sum": {k: float(v) for k, v in self._lrmuon_proj_ms.items()},
            "breakdown_ms": dict(getattr(self, "_step_breakdown", {}) or {}),   # ✅ 新增
        }
        self.timing_writer.write(rec)
    
    def consume_step_walltime_ms(self, ms: float):
        self._step_wall_ms = float(ms)  
        
    def consume_step_breakdown(self, breakdown: dict):
        if breakdown is None:
            return
        if not hasattr(self, "_step_breakdown") or self._step_breakdown is None:
            self._step_breakdown = {}
        # 用 float / int 的纯 python 类型，避免 JSON 序列化出问题
        for k, v in breakdown.items():
            if v is None:
                continue
            self._step_breakdown[k] = float(v) if isinstance(v, (int, float)) else v
    
    def get_stats_history(self):
        return self.stats_history

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
                param_names = group.get("param_names", None)
                
                for p in params:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    # FIX: don't rely on len(state)==0
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update, ns_ms = muon_update(
                        p.grad, state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group.get("ns_steps", 5),
                        nesterov=group.get("nesterov", True),
                        use_kimi_scaling=group.get("use_kimi_scaling", False),
                        kimi_c=group.get("kimi_c", 0.2),
                        timing=self.enable_timing
                    )
                    
                    self._muon_ns_ms += float(ns_ms)
                    
                    if update is None:
                        continue
                    if group.get("weight_decay", 0.0) != 0.0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

                # ---- (2) record ONCE per step ----
                do_record = (
                    self.enable_muon_svd_logging
                    and self.muon_stats_writer is not None
                    and self._cur_step is not None
                    and group.get("M_check_T", 0) > 0
                    and (int(self._cur_step) % int(group["M_check_T"]) == 0)
                    and _dist_rank() == 0
                )
                if do_record:
                    which = group.get("M_record_which", "buf")
                    rec = {
                        "step": int(self._cur_step),
                        "time": time.time(),
                        "optimizer": "muon",
                        "which": which,
                        "group": {
                            "momentum": float(group["momentum"]),
                            "ns_steps": int(group.get("ns_steps", 5)),
                            "nesterov": bool(group.get("nesterov", True)),
                        },
                        "modules": []
                    }

                    for idx, p2 in enumerate(params):
                        st = self.state.get(p2, None)
                        if not st or ("momentum_buffer" not in st):
                            continue
                        buf = st["momentum_buffer"]

                        if which == "M":
                            if p2.grad is None:
                                continue
                            M = p2.grad.add(buf, alpha=group["momentum"]) if group.get("nesterov", True) else buf
                            A = M
                        else:
                            A = buf

                        A2d = _flat2d(A).to(torch.float32)
                        s = torch.linalg.svdvals(A2d)

                        name = None
                        if param_names is not None and idx < len(param_names):
                            name = param_names[idx]

                        rec["modules"].append({
                            "name": name if name is not None else f"<muon_param_{idx}>",
                            "shape": [int(A2d.shape[0]), int(A2d.shape[1])],
                            "numel": int(A.numel()),
                            "sv": s.cpu().tolist(),
                        })

                    self.muon_stats_writer.write(rec)

                # ---- (3) optional broadcast ----
                if ws > 1 and group.get("distributed_broadcast", False):
                    for p in params:
                        dist.broadcast(p.data, src=0)
                


            elif group.get("use_lrmuon", False):
                
                
                params = group["params"]

                per_mod_stats = []  # list of dicts for this group
                number = 0
                
                method = str(group.get("lowrank_method", "svd")).lower()
                
                for p in params:
                    if p.grad is None:
                        continue
                    number += 1
                    state = self.state[p]

                    state["_timing_enabled"] = bool(self.enable_timing)
                    
                    stat = lrmuon_update(
                        p, p.grad, state,
                        lr=group["lr"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        rank_k=int(group["rank_k"]),
                        refresh_T=int(group["refresh_T"]),
                        ns_steps=int(group["ns_steps"]),
                        weight_decay=group.get("weight_decay", 0.0),
                        weight_norm_stabilize=group.get("weight_norm_stabilize", False),
                        use_kimi_scaling=group.get("use_kimi_scaling", False),
                        kimi_c=group.get("kimi_c", 0.2),
                        muon_eps=group.get("muon_eps", 1e-7),
                        number=number,
                        return_stats=self.enable_lrmuon_metrics,
                        lowrank_method=group.get("lowrank_method", "svd"),
                        lowrank_opt_steps=group.get("lowrank_opt_steps", 5),
                        lowrank_opt_lr=group.get("lowrank_opt_lr", 1e-2),
                    )
                    
                    if self.enable_timing:
                        self._lrmuon_ns_ms += float(state.get("lrmuon_last_ns_ms", 0.0))
                        proj_ms = float(state.get("lrmuon_last_proj_ms", 0.0))
                        if proj_ms > 0.0:  
                            self._lrmuon_proj_ms[method] = self._lrmuon_proj_ms.get(method, 0.0) + proj_ms
                    
                    
                    if stat is not None:
                        per_mod_stats.append(stat)

                if ws > 1 and group.get("distributed_broadcast", False):
                    for p in params:
                        dist.broadcast(p.data, src=0)
                
                
                # ===== DDP mean across ranks (per-module) =====
                # assume ordering consistent across ranks (params sorted by numel)
                if len(per_mod_stats) > 0 and (self._cur_step is not None):
                    device0 = params[0].device

                    # pack into tensors for all_reduce
                    # float tensors
                    P_ref = torch.tensor([s["P_refreshed"] for s in per_mod_stats], device=device0, dtype=torch.float32)
                    err_num = torch.stack([s["err_num"] for s in per_mod_stats]).to(torch.float32)
                    err_den = torch.stack([s["err_den"] for s in per_mod_stats]).to(torch.float32)
                    err     = torch.stack([s["err"] for s in per_mod_stats]).to(torch.float32)

                    # int tensors (use float32 for reduction; values should match across ranks)
                    M_rows = torch.tensor([s["M_rows"] for s in per_mod_stats], device=device0, dtype=torch.float32)
                    M_cols = torch.tensor([s["M_cols"] for s in per_mod_stats], device=device0, dtype=torch.float32)
                    P_k    = torch.tensor([s["P_k"]    for s in per_mod_stats], device=device0, dtype=torch.float32)
                    P_rows = torch.tensor([s["P_rows"] for s in per_mod_stats], device=device0, dtype=torch.float32)
                    
                    # 
                    
                    sign_err_num = torch.stack([s["sign_err_num"] for s in per_mod_stats]).to(torch.float32)
                    sign_err_den = torch.stack([s["sign_err_den"] for s in per_mod_stats]).to(torch.float32)
                    sign_err     = torch.stack([s["sign_err"]     for s in per_mod_stats]).to(torch.float32)

                    if ws > 1:
                        for t in (P_ref, err_num, err_den, err,
                                  sign_err_num, sign_err_den, sign_err,
                                  M_rows, M_cols, P_k, P_rows):
                            dist.all_reduce(t, op=dist.ReduceOp.SUM)
                        inv = 1.0 / float(ws)
                        for t in (P_ref, err_num, err_den, err,
                                  sign_err_num, sign_err_den, sign_err,
                                  M_rows, M_cols, P_k, P_rows):
                            t.mul_(inv)

                    # move to CPU python types
                    step_key = int(self._cur_step)
                    # 你可以把 group 信息也存进去（比如 rank_k/refresh_T）
                    payload = {
                        "group": {
                            "rank_k": int(group["rank_k"]),
                            "refresh_T": int(group["refresh_T"]),
                            "ns_steps": int(group["ns_steps"]),
                        },
                        "modules": []
                    }
                    for i in range(len(per_mod_stats)):
                        payload["modules"].append({
                            "P_refreshed": float(P_ref[i].item()),
                            "M_shape": (int(round(M_rows[i].item())), int(round(M_cols[i].item()))),
                            "P_shape": (int(round(P_k[i].item())), int(round(P_rows[i].item()))),  # (k, rows)
                            "err_num": float(err_num[i].item()),
                            "err_den": float(err_den[i].item()),
                            "err": float(err[i].item()),
                            "sign_err_num": float(sign_err_num[i].item()),
                            "sign_err_den": float(sign_err_den[i].item()),
                            "sign_err": float(sign_err[i].item()),
                        })

                    # store (note: if you have multiple lrmuon groups, you may want a list)
                    self.stats_history[step_key] = payload
                
                if self.stats_writer is not None and self._cur_step is not None and len(per_mod_stats) > 0:
                    record = {
                        "step": int(self._cur_step),
                        "time": time.time(),
                        "optimizer": "lrmuon",
                        **payload,   # payload 是 DDP mean 后的结果
                    }
                    self.stats_writer.write(record)
                

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
        
        
        
        
        # if (self.enable_timing and self.timing_writer is not None and self._cur_step is not None
        #     and (int(self._cur_step) % int(self.timing_check_T) == 0) and _dist_rank() == 0):
        #     rec = {
        #         "step": int(self._cur_step),
        #         "time": time.time(),
        #         "step_wall_ms": (None if self._step_wall_ms is None else float(self._step_wall_ms)),
        #         "muon_ns_ms_sum": float(self._muon_ns_ms),
        #         "lrmuon_ns_ms_sum": float(self._lrmuon_ns_ms),
        #         "lrmuon_proj_ms_by_method_sum": {k: float(v) for k, v in self._lrmuon_proj_ms.items()},
        #     }
        #     self.timing_writer.write(rec)
        
        
        return loss

    
    @torch.no_grad()
    def step_split_v0(self, forget_grads: dict, retain_grads: dict, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ws = _dist_world_size()

        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            params = group["params"]
            for p in params:
                gf = forget_grads.get(id(p), None)
                gr = retain_grads.get(id(p), None)
                if gf is None and gr is None:
                    continue

                state = self.state[p]
                if "momentum_buffer_forget" not in state:
                    state["momentum_buffer_forget"] = torch.zeros_like(p)
                if "momentum_buffer_retain" not in state:
                    state["momentum_buffer_retain"] = torch.zeros_like(p)

                upd_f, ns_ms_f = muon_update(
                    gf, state["momentum_buffer_forget"],
                    beta=group["momentum"],
                    ns_steps=group.get("ns_steps", 5),
                    nesterov=group.get("nesterov", True),
                    use_kimi_scaling=group.get("use_kimi_scaling", False),
                    kimi_c=group.get("kimi_c", 0.2),
                    timing=self.enable_timing
                )
                upd_r, ns_ms_r = muon_update(
                    gr, state["momentum_buffer_retain"],
                    beta=group["momentum"],
                    ns_steps=group.get("ns_steps", 5),
                    nesterov=group.get("nesterov", True),
                    use_kimi_scaling=group.get("use_kimi_scaling", False),
                    kimi_c=group.get("kimi_c", 0.2),
                    timing=self.enable_timing
                )

                self._muon_ns_ms += float(ns_ms_f) + float(ns_ms_r)

                if upd_f is None and upd_r is None:
                    continue
                if upd_f is None:
                    update = upd_r
                elif upd_r is None:
                    update = upd_f
                else:
                    update = upd_f + upd_r

                if group.get("weight_decay", 0.0) != 0.0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

            if ws > 1 and group.get("distributed_broadcast", False):
                for p in params:
                    dist.broadcast(p.data, src=0)

        return loss

    # -------------------------
    # v1: Eq.20 split (msign([Ur,Uf]) msign([Vr,Vf])^T)
    # -------------------------
    @torch.no_grad()
    def step_split_v1(self, forget_grads: dict, retain_grads: dict, closure=None):
        # 这里调用我上一条给你的 muon_update_v1_split / msign_via_svd / svd_subspace
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ws = _dist_world_size()

        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            params = group["params"]
            param_names = group.get("param_names", None)

            for idx, p in enumerate(params):
                gf = forget_grads.get(id(p), None)
                gr = retain_grads.get(id(p), None)
                if gf is None and gr is None:
                    continue

                state = self.state[p]
                if "momentum_buffer_forget" not in state:
                    state["momentum_buffer_forget"] = torch.zeros_like(p)
                if "momentum_buffer_retain" not in state:
                    state["momentum_buffer_retain"] = torch.zeros_like(p)

                out = muon_update_v1_split_topk(
                    gf, gr,
                    state["momentum_buffer_forget"],
                    state["momentum_buffer_retain"],
                    beta=group["momentum"],
                    nesterov=group.get("nesterov", True),
                    weight_decay=group.get("weight_decay", 0.0),
                    lr=group["lr"],
                    v1_k=int(group.get("v1_k", 256)),
                    use_kimi_scaling=group.get("use_kimi_scaling", False),
                    kimi_c=group.get("kimi_c", 0.2),
                    enforce_total_rank_cap=bool(group.get("enforce_total_rank_cap", True)),
                )

                if out is None:
                    continue

                # update, sti_before, sti_after = out
                
                update= out

                # apply update
                if group.get("weight_decay", 0.0) != 0.0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

                # # ===== write STI record (only for your 3 target params) =====
                # if (self.sti_writer is not None) and (self._cur_step is not None) and (_dist_rank() == 0):
                #     name = None
                #     if param_names is not None and idx < len(param_names):
                #         name = param_names[idx]
                #     else:
                #         name = f"<muon_param_{idx}>"

                #     if name in TARGET_PARAM_NAME_SET:
                #         rec = {
                #             "step": int(self._cur_step),
                #             "time": time.time(),
                #             "param_name": name,
                #             "shape": [int(p.shape[0]), int(p.shape[1])] if p.ndim == 2 else list(p.shape),
                #             "sti_before": float(sti_before.item()),
                #             "sti_after": float(sti_after.item()),
                #             "v1_k": int(group.get("v1_k", 256)),
                #         }
                #         self.sti_writer.write(rec)  # JSONL, 即时写入

            if ws > 1 and group.get("distributed_broadcast", False):
                for p in params:
                    dist.broadcast(p.data, src=0)

        return loss
    
    @torch.no_grad()
    def step_split_v2(self, forget_grads: dict, retain_grads: dict, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ws = _dist_world_size()

        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            # [NEW] external k (common subspace rank)
            v2_common_k = int(group.get("v2_common_k", 256))

            params = group["params"]
            for p in params:
                gf = forget_grads.get(id(p), None)
                gr = retain_grads.get(id(p), None)

                state = self.state[p]
                if "momentum_buffer_common" not in state:
                    state["momentum_buffer_common"] = torch.zeros_like(p)
                if "momentum_buffer_forget" not in state:
                    state["momentum_buffer_forget"] = torch.zeros_like(p)
                if "momentum_buffer_retain" not in state:
                    state["momentum_buffer_retain"] = torch.zeros_like(p)

                update = muon_update_v2_split_leftU(
                    gf, gr,
                    state["momentum_buffer_common"],
                    state["momentum_buffer_forget"],
                    state["momentum_buffer_retain"],
                    beta=group["momentum"],
                    nesterov=group.get("nesterov", True),
                    weight_decay=group.get("weight_decay", 0.0),
                    lr=group["lr"],
                    v2_common_k=v2_common_k,
                    use_kimi_scaling=group.get("use_kimi_scaling", False),
                    kimi_c=group.get("kimi_c", 0.2),
                )
                if update is None:
                    continue

                if group.get("weight_decay", 0.0) != 0.0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

            if ws > 1 and group.get("distributed_broadcast", False):
                for p in params:
                    dist.broadcast(p.data, src=0)

        return loss
    
    @torch.no_grad()
    def step_split_v0_topk(self, forget_grads: dict, retain_grads: dict, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ws = _dist_world_size()

        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            params = group["params"]
            param_names = group.get("param_names", None)

            for idx, p in enumerate(params):
                gf = forget_grads.get(id(p), None)
                gr = retain_grads.get(id(p), None)
                if gf is None and gr is None:
                    continue

                state = self.state[p]
                if "momentum_buffer_forget" not in state:
                    state["momentum_buffer_forget"] = torch.zeros_like(p)
                if "momentum_buffer_retain" not in state:
                    state["momentum_buffer_retain"] = torch.zeros_like(p)

                out = muon_update_v0_svd_concat(
                    gf, gr,
                    state["momentum_buffer_forget"],
                    state["momentum_buffer_retain"],
                    beta=group["momentum"],
                    nesterov=group.get("nesterov", True),
                    weight_decay=group.get("weight_decay", 0.0),
                    lr=group["lr"],
                    v0_k=int(group.get("v0_k", 256)),
                    use_kimi_scaling=group.get("use_kimi_scaling", True),
                    kimi_c=group.get("kimi_c", 0.2),
                    enforce_total_rank_cap=bool(group.get("enforce_total_rank_cap", True)),
                    return_uvhat=True,
                )

                if out is None:
                    continue

                update, sti_before = out

                if group.get("weight_decay", 0.0) != 0.0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

                # ===== write STI record (rank0, only target params) =====
                if (self.sti_writer is not None) and (self._cur_step is not None) and (_dist_rank() == 0):
                    if param_names is not None and idx < len(param_names):
                        name = param_names[idx]
                    else:
                        name = f"<muon_param_{idx}>"

                    if name in TARGET_PARAM_NAME_SET:
                        self.sti_writer.write({
                            "step": int(self._cur_step),
                            "time": time.time(),
                            "mode": "v0-topk",
                            "param_name": name,
                            "shape": [int(p.shape[0]), int(p.shape[1])] if p.ndim == 2 else list(p.shape),
                            "v0_k": int(group.get("v0_k", 256)),
                            "sti_before": float(sti_before.item()),
                        })

            # broadcast 应该在“这个 group 的 params”处理完之后
            if ws > 1 and group.get("distributed_broadcast", False):
                for p in params:
                    dist.broadcast(p.data, src=0)

        return loss
    
    # -------------------------
    # unified entry: keep your main file unchanged if you want
    # -------------------------
    # @torch.no_grad()
    # def step_split(self, forget_grads: dict, retain_grads: dict, closure=None, mode: str = "v0"):
    #     mode = (mode or "v0").lower()
    #     if mode == "v0":
    #         return self.step_split_v0(forget_grads, retain_grads, closure=closure)
    #     elif mode == "v1":
    #         return self.step_split_v1(forget_grads, retain_grads, closure=closure)
    #     elif mode == "v2":
    #         return self.step_split_v2(forget_grads, retain_grads, closure=closure)
    #     elif mode == "v0-topk":
    #         return self.step_split_v0_topk(forget_grads, retain_grads, closure=closure)
    #     else:
    #         raise ValueError(f"Unknown mode for step_split: {mode}")
    
    @torch.no_grad()
    def step_split(
        self,
        forget_grads: dict,
        retain_grads: dict,
        closure=None,
        mode: str = "v0",
        *,
        switch_after_k: int | None = None,     # NEW
        v1_mode: str = "v1",                   # NEW (默认前期用 v1)
        v0_mode: str = "v0-topk",              # NEW (默认后期用 v0-topk)
    ):
        """
        mode:
        - "v0", "v1", "v2", "v0-topk" : keep old behavior
        - "auto" : if self._cur_step < switch_after_k -> v1_mode else v0_mode
        """
        mode = (mode or "v0-topk").lower()

        print("now we use  the mode .. ",mode)
        # import sys
        # sys.exit()
        
        if mode == "auto":
            k = 0 if switch_after_k is None else int(switch_after_k)
            cur = 0 if self._cur_step is None else int(self._cur_step)

            chosen = (v1_mode if cur < k else v0_mode)
            chosen = (chosen or "v0-topk").lower()

            if chosen == "v0":
                print("v0")
                return self.step_split_v0(forget_grads, retain_grads, closure=closure)
            elif chosen == "v1":
                print("v1")
                return self.step_split_v1(forget_grads, retain_grads, closure=closure)
            elif chosen == "v2":
                return self.step_split_v2(forget_grads, retain_grads, closure=closure)
            elif chosen == "v0-topk":
                print("v0-topk")
                return self.step_split_v0_topk(forget_grads, retain_grads, closure=closure)
            else:
                raise ValueError(f"Unknown chosen mode in auto: {chosen}")

        # ---- original dispatch ----
        if mode == "v0":
            return self.step_split_v0(forget_grads, retain_grads, closure=closure)
        elif mode == "v1":
            return self.step_split_v1(forget_grads, retain_grads, closure=closure)
        elif mode == "v2":
            return self.step_split_v2(forget_grads, retain_grads, closure=closure)
        elif mode == "v0-topk":
            return self.step_split_v0_topk(forget_grads, retain_grads, closure=closure)
        else:
            raise ValueError(f"Unknown mode for step_split: {mode}")

    
    
def build_muon_param_groups(model, configs):
    # --- enforce exclusivity with a seen set (by object id) ---
    seen = set()
    adam_only_regexes = getattr(configs, "adam_only_param_regexes", [r"(?:^|\.)(lm_head)(?:\.|$)"])
    include_convs = bool(getattr(configs, "include_convs_in_muon", False))

    # NEW: id(param) -> full name mapping
    id2name = {id(p): name for name, p in model.named_parameters()}
    
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
    
    
    # NEW: names aligned with muon_params order
    muon_param_names = [id2name.get(id(p), "<unnamed>") for p in muon_params]
    
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
            lowrank_method=configs.lowrank_method,
            lowrank_opt_steps=configs.lowrank_opt_steps,
            lowrank_opt_lr=configs.lowrank_opt_lr,
        )
        return [*adam_groups, lrmuon_group]
    else:
        muon_group = dict(
            params=muon_params,
            lr=configs.muon_lr,
            param_names=muon_param_names,  # NEW
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
            M_stats_path=getattr(configs, "M_stats_path", ""),
            M_check_T=int(getattr(configs, "M_check_T", 0)),
            M_record_which=str(getattr(configs, "M_record_which", "buf")),
            v2_common_k=int(getattr(configs, "V2_COMMON_K", 256)),
            
            v0_k=int(getattr(configs, "V0_K", 256)),
            enforce_total_rank_cap=bool(getattr(configs, "V0_CAP", True)),
            v1_k=int(getattr(configs, "v1_k", 256)),
            STI_path=str(getattr(configs, "STI_path", "")),
            
        )
        return [*adam_groups, muon_group]

