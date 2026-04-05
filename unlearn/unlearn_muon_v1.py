import os
import datetime

import numpy as np
import torch
import tqdm as tqdm
import wandb

from rmu.utils import load_model, get_params, forward_with_cache, get_data
from dataclasses import dataclass

from rmu_wb.muon import build_muon_param_groups, MuonWithAuxAdam


@dataclass
class MuonConfig:
    """Configuration for Muon optimizer matching your muon.py structure"""
    # Muon (full) parameters
    muon_lr: float = 0.016
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    muon_max_grad_norm: float = 1.0
    
    # Adam side (Embeddings/bias/scalars)
    adam_emb_lr: float = 0.0032
    adam_scalar_lr: float = 0.0032
    adam_eps_small: float = 1e-15
    adam_betas_muon_side: tuple = (0.8, 0.98)
    adam_weight_decay: float = 0.8
    adam_max_grad_norm: float = 1.0
    
    # Routing/masking
    adam_only_param_regexes: tuple = (r'(?:^|\.)(lm_head)(?:\.|$)',)
    include_convs_in_muon: bool = False
    
    # Common settings
    distributed_broadcast: bool = False
    muon_use_kimi_scaling: bool = False
    muon_kimi_c: float = 0.2
    muon_eps: float = 1e-7
    
    v1_k: int = 256  # NEW
    enforce_total_rank_cap: bool = True  # NEW
    
    # Low-rank Muon (not used in standard muon)
    use_low_rank_muon: bool = False
    
    # Logging
    M_stats_path: str = ""
    M_check_T: int = 0
    M_record_which: str = "buf"
    enable_muon_svd_logging: bool = False
    enable_lrmuon_metrics: bool = False
    enable_timing: bool = False
    timing_path: str = ""
    timing_check_T: int = 1

def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    rmu_config = vars(args)
    print("====rmu Config====")
    print("\n".join(f"{k}={v}" for k,v in rmu_config.items()))
    print("=====")

    # Initialize wandb
    model_name = args.model_name_or_path.split('/')[-1]
    wandb.init(
        project="Muon-constraint",
        name=f"muon-v1-rank-{args.output_dir}",
        config=rmu_config
    )

    updated_model = updated_model.train()
    
    picked_ids = set(id(p) for p in get_params(updated_model, args.layer_ids, args.param_ids))
    
    # Build Muon config from args
    muon_cfg = MuonConfig(
        muon_lr=args.muon_lr,
        muon_momentum=getattr(args, 'muon_momentum', 0.95),
        muon_weight_decay=getattr(args, 'muon_weight_decay', 0.1),
        muon_ns_steps=getattr(args, 'muon_ns_steps', 5),
        muon_nesterov=getattr(args, 'muon_nesterov', True),
        adam_emb_lr=getattr(args, 'adam_emb_lr', 0.0032),
        adam_scalar_lr=getattr(args, 'adam_scalar_lr', 0.0032),
        enable_muon_svd_logging=getattr(args, 'enable_muon_svd_logging', False),
        M_stats_path=getattr(args, 'M_stats_path', ""),
        M_check_T=getattr(args, 'M_check_T', 0),
        v1_k=int(getattr(args, "V1_K", 256)),                     # NEW
        # enforce_total_rank_cap=bool(getattr(configs, "V1_CAP", True))# NEW
        enforce_total_rank_cap=True
    )
    
    # Create optimizer using MuonWithAuxAdam
    stats_path = muon_cfg.M_stats_path if muon_cfg.M_stats_path else None
    timing_path = muon_cfg.timing_path if muon_cfg.timing_path else None
    
    param_groups = build_muon_param_groups(updated_model, muon_cfg)
    
    new_param_groups = []
    for g in param_groups:
        params = g["params"]
        keep_mask = [id(p) in picked_ids for p in params]
        new_params = [p for p, keep in zip(params, keep_mask) if keep]
        if len(new_params) == 0:
            continue

        pg = dict(g)
        pg["params"] = new_params

        # 同步裁剪 muon 组的 param_names（如果存在）
        if "param_names" in pg:
            names = pg["param_names"]
            pg["param_names"] = [n for n, keep in zip(names, keep_mask) if keep]

        new_param_groups.append(pg)

    param_groups = new_param_groups
    
    optimizer = MuonWithAuxAdam(
        param_groups,
        stats_path=stats_path,
        timing_path=timing_path,
        enable_muon_svd_logging=muon_cfg.enable_muon_svd_logging,
        enable_lrmuon_metrics=muon_cfg.enable_lrmuon_metrics,
        timing_check_T=muon_cfg.timing_check_T,
    )
    
    picked = get_params(updated_model, args.layer_ids, args.param_ids)
    id2name = {id(p): n for n, p in updated_model.named_parameters()}
    
    # 先建立 id -> route 的字典
    id2route = {}
    for g in param_groups:
        route = "MUON" if g.get("use_muon", False) else ("LRMUON" if g.get("use_lrmuon", False) else "ADAM")
        for p in g["params"]:
            id2route[id(p)] = route

    # 打印 picked 的路由
    for p in picked:
        pid = id(p)
        print(id2route.get(pid, "NOT_IN_OPTIMIZER"), id2name.get(pid, "<unnamed>"), tuple(p.shape))
    
    picked = set(get_params(updated_model, args.layer_ids, args.param_ids))
    for p in updated_model.parameters():
        p.requires_grad_(False)
    for p in picked:
        p.requires_grad_(True)
    
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(1, 1, updated_model.config.hidden_size, 
                                   dtype=updated_model.dtype, device=updated_model.device)
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )
    
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                control_vec = control_vectors_list[topic_idx]
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Unlearning loss
                max_length = 512 if topic_idx == 0 else 768
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                )
                unlearn_inputs = unlearn_inputs.to(updated_model.device)
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec
                )

                # Retain loss
                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(updated_model.device)
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= args.alpha[topic_idx]

                # ===== split grads =====
                optimizer.zero_grad(set_to_none=True)

                # (1) forget grads
                unlearn_loss.backward(retain_graph=True)
                forget_grads = {}
                for p in picked:  # picked is a set of params
                    if p.grad is not None:
                        forget_grads[id(p)] = p.grad.detach().clone()

                optimizer.zero_grad(set_to_none=True)

                # (2) retain grads
                retain_loss.backward()
                retain_grads = {}
                for p in picked:
                    if p.grad is not None:
                        retain_grads[id(p)] = p.grad.detach().clone()

                optimizer.zero_grad(set_to_none=True)

                # (3) optimizer step with split-momentum
                optimizer.set_step(idx)
                # optimizer.step_split(forget_grads, retain_grads)
                
                optimizer.step_split(forget_grads, retain_grads, mode="v1")
                
                loss = unlearn_loss + retain_loss

                # Calculate param change (from first param group that has gradients)
                param_change = 0.0
                for param in params:
                    if param.grad is not None:
                        param_change = param.grad.abs().mean().item()
                        break
                
                # Log to wandb
                wandb.log({
                    "loss": loss.item(),
                    "unlearn_loss": unlearn_loss.item(),
                    "retain_loss": retain_loss.item(),
                    "param_change": param_change,
                    "step": idx
                })
                
                print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {param_change:.4g}")
                
                # ======= Logging ======
                if args.verbose:
                    frozen_forget_activations = forward_with_cache(
                        frozen_model, unlearn_inputs, module=frozen_module, no_grad=True
                    ).to(updated_model.device)
                    unlearn_cosine = torch.nn.functional.cosine_similarity(
                        updated_forget_activations, frozen_forget_activations, dim=-1
                    ).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(
                        updated_retain_activations, frozen_retain_activations, dim=-1
                    ).mean()
                    
                    wandb.log({
                        "unlearn_cosine_sim": unlearn_cosine.item(),
                        "retain_cosine_sim": retain_cosine.item(),
                        "step": idx
                    })
                    
                    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                    print(f"retain_cosine_sim={retain_cosine.item()}")
                    print(f"Topic {topic_idx} updated_forget_activations.norm=",
                          torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} frozen_forget_activations.norm=",
                          torch.mean(frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} updated_retain_activations.norm=",
                          torch.mean(updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} frozen_retain_activations.norm=",
                          torch.mean(frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())

                pbar.update(1)
                
                # Write timing record if enabled
                if hasattr(optimizer, "write_timing_record"):
                    optimizer.write_timing_record()

    tokenizer.truncation_side = truncation_side
    
    # Save model
    if args.output_dir:
        path = args.output_dir
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"models/{args.model_name_or_path}_muon_lr-{args.muon_lr}_alpha-{args.alpha}_batches-{num_batches}_layer-{args.layer_id}_{date}"
    
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")
    
    # Save path to model_dir.txt
    save_model_path_to_file(path, "/egr/research-optml/wangc168/Muon_wmdp/rmu_wmdp/model_dir.txt")
    
    wandb.finish()


def save_model_path_to_file(model_path, file_path):
    """Save model path with incrementing index to file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Read last index
    last_index = 0
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line:
                    try:
                        # Extract number from start of line
                        last_index = int(last_line.split()[0])
                    except (ValueError, IndexError):
                        pass
    
    # Write new entry
    new_index = last_index + 1
    with open(file_path, 'a') as f:
        f.write(f"{new_index} {model_path}\n")
    
    print(f"Saved model path to {file_path} with index {new_index}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta"
    )
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.model.layers[{layer_id}]"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    ### Data arguments
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default="wikitext,wikitext",
        help="comma-separated list of corpora to retain",
    )
    parser.add_argument(
        "--forget_corpora",
        type=str,
        default="bio-forget-corpus,cyber-forget-corpus",
        help="comma-separated list of corpora to forget",
    )
    ### RMU hyperparameters
    parser.add_argument("--alpha", type=str, default="100,100", help="retain weight")
    parser.add_argument(
        "--steering_coeffs",
        type=str,
        default="20,20",
        help="Steer vector weight in order of topic",
    )
    
    ### Muon optimizer parameters
    parser.add_argument("--muon_lr", type=float, default=0.016, help="Muon learning rate")
    parser.add_argument("--muon_momentum", type=float, default=0.95, help="Muon momentum")
    parser.add_argument("--muon_weight_decay", type=float, default=0.1, help="Muon weight decay")
    parser.add_argument("--muon_ns_steps", type=int, default=5, help="Newton-Schulz steps")
    parser.add_argument("--muon_nesterov", type=bool, default=True, help="Use Nesterov momentum")
    
    ### Adam side parameters
    parser.add_argument("--adam_emb_lr", type=float, default=0.0032, help="Adam embedding lr")
    parser.add_argument("--adam_scalar_lr", type=float, default=0.0032, help="Adam scalar lr")
    
    ### Logging parameters
    parser.add_argument("--enable_muon_svd_logging", action="store_true", help="Enable Muon SVD logging")
    parser.add_argument("--M_stats_path", type=str, default="", help="Path for Muon stats")
    parser.add_argument("--M_check_T", type=int, default=0, help="Check interval for Muon stats")
    
    parser.add_argument(
        "--V1_K",
        type=int,
        default=256,
        help="Top-k truncation rank for Muon split v1 (Ur/Uf, Vr/Vf use top-k).",
    )

    
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")

    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args 


if __name__ == "__main__":
    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )