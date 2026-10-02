#!/usr/bin/env python3
"""Fine-tune MACELES-uiu on JARVIS Born effective charge labels.

This is a diagnostic transfer-learning experiment. It uses JARVIS DFPT BEC
labels to directly supervise the model's ``compute_bec=True`` output, then saves
the fine-tuned model for downstream ionic-piezo tests.
"""

import argparse
import concurrent.futures
import csv
import gc
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from mace.tools.utils import AtomicNumberTable


os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-maceles-uiu")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

DEFAULT_MODEL = (
    "/home/liaohui/桌面/MACELES-MPtrj-uiu/"
    "checkpoints/MACELES-MPtrj-uiu-r5.5_run-123_stagetwo.model"
)
DEFAULT_JSONL = "/home/liaohui/桌面/mace-les/mace-main/jarvis_dfpt_extracted.jsonl"
DEFAULT_OUTDIR = "results/uiu_bec_jarvis_finetune"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune MACELES-uiu compute_bec=True output on JARVIS DFPT BEC labels."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument(
        "--valid_jsonl",
        default=None,
        help=(
            "Optional explicit validation JSONL. When set, --jsonl is used as "
            "the training set and valid_fraction is ignored."
        ),
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Optional comma-separated synchronous data-parallel devices, for "
            "example cuda:0,cuda:1. --batch_size remains the global batch size."
        ),
    )
    parser.add_argument("--default_dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--valid_fraction", type=float, default=0.2)
    parser.add_argument("--max_structures", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--loss", choices=["huber", "mse", "mae"], default="huber")
    parser.add_argument("--huber_beta", type=float, default=0.2)
    parser.add_argument("--asr_weight", type=float, default=0.05)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument(
        "--randomize_parameters",
        choices=["none", "per_tensor_normal"],
        default="none",
        help=(
            "Randomize floating-point model parameters after loading the architecture. "
            "Use this for BEC-supervised from-scratch experiments; the loaded model then "
            "acts only as an architecture/elements template."
        ),
    )
    parser.add_argument(
        "--train_scope",
        choices=["les_readouts", "les_and_lora", "les_and_last_interaction", "all"],
        default="les_readouts",
    )
    parser.add_argument("--eval_every", type=int, default=1)
    return parser.parse_args()


def set_default_dtype(name):
    if name == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_args(args):
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def devices_from_args(args):
    if args.devices is None:
        return [device_from_args(args)]
    devices = [torch.device(item.strip()) for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices did not contain any device")
    if len(set(map(str, devices))) != len(devices):
        raise ValueError(f"--devices contains duplicates: {devices}")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable")
    return devices


def ase_atoms_from_jarvis_record(record):
    structure = record["structure"]
    cell = np.asarray(structure["lattice"]["matrix"], dtype=float)
    symbols = []
    positions = []
    for site in structure["sites"]:
        species = site["species"]
        if len(species) != 1:
            raise ValueError("mixed-occupancy JARVIS site is not supported")
        symbols.append(species[0]["element"])
        positions.append(site["xyz"])
    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=cell,
        pbc=True,
    )
    atoms.info["jid"] = record.get("jid", "")
    atoms.info["source_dir"] = record.get("source_dir", "")
    if "energy_eV" in record:
        atoms.info["REF_energy"] = float(record["energy_eV"])
    if "stress_kbar" in record:
        # Keep the raw value for metadata; this script does not train stress.
        atoms.info["stress_kbar"] = np.asarray(record["stress_kbar"], dtype=float).reshape(3, 3)
    return atoms


def load_jarvis_examples(args, model, jsonl_path=None, max_structures=None):
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers.cpu().numpy()])
    supported = set(z_table.zs)
    cutoff = float(model.r_max.detach().cpu().item())
    jsonl_path = jsonl_path or args.jsonl
    max_structures = args.max_structures if max_structures is None else max_structures

    examples = []
    skipped = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_structures is not None and len(examples) >= max_structures:
                break
            try:
                record = json.loads(line)
                if "born_effective_charge" not in record:
                    raise ValueError("missing born_effective_charge")
                atoms = ase_atoms_from_jarvis_record(record)
                unsupported = sorted(set(int(z) for z in atoms.get_atomic_numbers()) - supported)
                if unsupported:
                    raise ValueError(f"unsupported Z: {unsupported}")
                bec = np.asarray(record["born_effective_charge"], dtype=np.float32).reshape(-1, 3, 3)
                if bec.shape[0] != len(atoms):
                    ratio = bec.shape[0] / max(len(atoms), 1)
                    raise ValueError(
                        "BEC atom count mismatch: "
                        f"structure_natoms={len(atoms)}, bec_natoms={bec.shape[0]}, "
                        f"ratio={ratio:.3g}. This record likely mixes primitive/conventional "
                        "cell representations, so it is skipped by default."
                    )
                data = AtomicData.from_config(config_from_atoms(atoms), z_table=z_table, cutoff=cutoff)
                examples.append(
                    {
                        "index": idx,
                        "jsonl": str(jsonl_path),
                        "jid": record.get("jid", ""),
                        "formula": atoms.get_chemical_formula(),
                        "natoms": len(atoms),
                        "target_bec": bec,
                        "data": data,
                    }
                )
            except Exception as exc:
                skipped.append({"jsonl": str(jsonl_path), "index": idx, "reason": str(exc)})
    return examples, skipped


def split_examples(examples, valid_fraction, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    n_valid = max(1, int(round(len(indices) * valid_fraction)))
    valid_idx = set(indices[:n_valid].tolist())
    train, valid = [], []
    for i, example in enumerate(examples):
        (valid if i in valid_idx else train).append(example)
    return train, valid


def iter_minibatches(items, batch_size, shuffle, seed):
    indices = list(range(len(items)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [items[i] for i in indices[start : start + batch_size]]


def batch_to_device(items, device):
    loader = torch_geometric.dataloader.DataLoader(
        [item["data"] for item in items],
        batch_size=len(items),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to(device)
    target = torch.as_tensor(
        np.concatenate([item["target_bec"] for item in items], axis=0),
        dtype=torch.get_default_dtype(),
        device=device,
    )
    return batch, target


def normalize_bec_output(raw_bec, n_atoms):
    if raw_bec.ndim == 2 and tuple(raw_bec.shape) == (n_atoms, 9):
        return raw_bec.reshape(n_atoms, 3, 3)
    if raw_bec.ndim == 3 and tuple(raw_bec.shape) == (n_atoms, 3, 3):
        return raw_bec
    if raw_bec.ndim == 4 and raw_bec.shape[0] == n_atoms and raw_bec.shape[-2:] == (3, 3):
        return raw_bec.reshape(n_atoms, -1, 3, 3).sum(dim=1)
    if raw_bec.ndim == 4 and raw_bec.shape[1] == n_atoms and raw_bec.shape[-2:] == (3, 3):
        return raw_bec.reshape(-1, n_atoms, 3, 3).sum(dim=0)
    raise ValueError(f"unexpected BEC output shape {tuple(raw_bec.shape)}")


def predict_bec(model, batch, training):
    output = model(
        batch.to_dict(),
        training=training,
        compute_force=False,
        compute_bec=True,
    )
    if output.get("BEC") is None:
        raise RuntimeError("model output does not contain BEC")
    return normalize_bec_output(output["BEC"], batch.num_nodes)


def elementwise_loss(pred, target, args):
    if args.loss == "mse":
        return torch.nn.functional.mse_loss(pred, target)
    if args.loss == "mae":
        return torch.nn.functional.l1_loss(pred, target)
    return torch.nn.functional.smooth_l1_loss(pred, target, beta=args.huber_beta)


def graphwise_bec_loss(pred, target, ptr, args):
    losses = []
    for graph_idx in range(ptr.numel() - 1):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        losses.append(elementwise_loss(pred[start:end], target[start:end], args))
    return torch.stack(losses).mean()


def asr_loss(pred, ptr):
    losses = []
    for graph_idx in range(ptr.numel() - 1):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        losses.append(torch.mean(torch.sum(pred[start:end], dim=0) ** 2))
    return torch.stack(losses).mean()


def split_global_batch(items, parts):
    indices = np.array_split(np.arange(len(items)), min(parts, len(items)))
    return [[items[int(index)] for index in chunk] for chunk in indices if len(chunk)]


def replica_backward(model, items, device, args, weight):
    model.train()
    batch, target = batch_to_device(items, device)
    pred = predict_bec(model, batch, training=True)
    bec_loss = graphwise_bec_loss(pred, target, batch.ptr, args)
    zero_sum_loss = asr_loss(pred, batch.ptr)
    loss = bec_loss + args.asr_weight * zero_sum_loss
    (float(weight) * loss).backward()
    result = (
        float(loss.detach().cpu()),
        float(bec_loss.detach().cpu()),
        float(zero_sum_loss.detach().cpu()),
        len(items),
    )
    del batch, target, pred, loss, bec_loss, zero_sum_loss
    return result


def synchronize_trainable_parameters(primary, replicas, trainable_names):
    primary_named = dict(primary.named_parameters())
    with torch.no_grad():
        for replica in replicas[1:]:
            replica_named = dict(replica.named_parameters())
            for name in trainable_names:
                replica_named[name].copy_(
                    primary_named[name].detach().to(replica_named[name].device)
                )


def synchronous_data_parallel_step(
    models,
    devices,
    items,
    args,
    optimizer,
    trainable_names,
    executor,
):
    chunks = split_global_batch(items, len(models))
    active_models = models[: len(chunks)]
    active_devices = devices[: len(chunks)]
    for model in active_models:
        model.zero_grad(set_to_none=True)
    futures = []
    for model, device, chunk in zip(active_models, active_devices, chunks):
        futures.append(
            executor.submit(
                replica_backward,
                model,
                chunk,
                device,
                args,
                len(chunk) / len(items),
            )
        )
    results = [future.result() for future in futures]
    primary_named = dict(models[0].named_parameters())
    for replica in active_models[1:]:
        replica_named = dict(replica.named_parameters())
        for name in trainable_names:
            gradient = replica_named[name].grad
            if gradient is None:
                continue
            target = primary_named[name]
            contribution = gradient.to(target.device)
            if target.grad is None:
                target.grad = contribution.clone()
            else:
                target.grad.add_(contribution)
    trainable = [primary_named[name] for name in trainable_names]
    if args.clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
    optimizer.step()
    synchronize_trainable_parameters(models[0], models, trainable_names)
    total = sum(result[3] for result in results)
    return tuple(
        sum(result[column] * result[3] for result in results) / total
        for column in range(3)
    )


def metrics_from_arrays(ref, pred):
    ref = np.asarray(ref, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    err = pred - ref
    if ref.size < 2 or np.std(ref) == 0.0 or np.std(pred) == 0.0:
        pearson = math.nan
    else:
        pearson = float(np.corrcoef(ref, pred)[0, 1])
    return {
        "n_values": int(ref.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "pearson": pearson,
        "ref_abs_mean": float(np.mean(np.abs(ref))),
        "pred_abs_mean": float(np.mean(np.abs(pred))),
    }


def evaluate(model, items, args, device, write_rows_path=None):
    model.eval()
    all_ref = []
    all_pred = []
    rows = []
    for batch_items in iter_minibatches(items, args.batch_size, shuffle=False, seed=args.seed):
        batch, target = batch_to_device(batch_items, device)
        pred = predict_bec(model, batch, training=False)
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        all_pred.append(pred_np)
        all_ref.append(target_np)

        start = 0
        for item in batch_items:
            end = start + item["natoms"]
            item_pred = pred_np[start:end]
            item_ref = target_np[start:end]
            m = metrics_from_arrays(item_ref, item_pred)
            rows.append(
                {
                    "index": item["index"],
                    "jid": item["jid"],
                    "formula": item["formula"],
                    "natoms": item["natoms"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "ref_abs_mean": m["ref_abs_mean"],
                    "pred_abs_mean": m["pred_abs_mean"],
                }
            )
            start = end
        del batch, target, pred
    metrics = metrics_from_arrays(np.concatenate(all_ref, axis=0), np.concatenate(all_pred, axis=0))
    if write_rows_path is not None:
        with open(write_rows_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "jid",
                    "formula",
                    "natoms",
                    "mae",
                    "rmse",
                    "ref_abs_mean",
                    "pred_abs_mean",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def set_trainable_parameters(model, scope):
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    les_prefixes = ("les_readouts", "les_u_readouts", "les_alpha_readouts")
    n_trainable = 0
    trainable_names = []
    for name, param in model.named_parameters():
        enabled = False
        if scope == "all":
            enabled = True
        elif scope == "les_readouts":
            enabled = name.startswith(les_prefixes)
        elif scope == "les_and_lora":
            enabled = name.startswith(les_prefixes) or ".LoRA_weight." in name
        elif scope == "les_and_last_interaction":
            enabled = name.startswith(les_prefixes) or name.startswith("interactions.1")
        if enabled:
            param.requires_grad_(True)
            n_trainable += param.numel()
            trainable_names.append(name)
    return n_trainable, trainable_names


def randomize_model_parameters(model, mode, seed):
    if mode == "none":
        return []

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    rows = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not torch.is_floating_point(param):
                continue
            before = param.detach().cpu().float()
            std = float(torch.std(before, unbiased=False))
            if not math.isfinite(std) or std < 1.0e-6:
                std = 0.02
            values = torch.randn(
                param.shape,
                generator=rng,
                dtype=torch.float32,
                device="cpu",
            ) * std
            param.copy_(values.to(device=param.device, dtype=param.dtype))
            rows.append(
                {
                    "name": name,
                    "shape": "x".join(str(dim) for dim in param.shape),
                    "init_mode": mode,
                    "std_used": std,
                }
            )
    return rows


def write_history(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_default_dtype(args.default_dtype)
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    devices = devices_from_args(args)
    device = devices[0]

    model = torch.load(args.model, map_location=device, weights_only=False)
    if model.__class__.__name__ != "MACELES":
        raise RuntimeError(f"expected MACELES model, got {model.__class__.__name__}")
    model.to(device)

    init_rows = randomize_model_parameters(model, args.randomize_parameters, args.seed)
    if init_rows:
        print(
            f"Randomized {len(init_rows)} floating-point parameter tensors "
            f"with mode={args.randomize_parameters}"
        )

    n_trainable, trainable_names = set_trainable_parameters(model, args.train_scope)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters selected")
    models = [model]
    for replica_device in devices[1:]:
        replica = torch.load(args.model, map_location=replica_device, weights_only=False)
        if init_rows:
            randomize_model_parameters(replica, args.randomize_parameters, args.seed)
        replica.to(replica_device)
        replica_trainable, replica_names = set_trainable_parameters(
            replica, args.train_scope
        )
        if replica_trainable != n_trainable or replica_names != trainable_names:
            raise RuntimeError("data-parallel replicas have different trainable parameters")
        models.append(replica)

    print(f"Loading JARVIS BEC train examples from {args.jsonl}")
    examples, skipped = load_jarvis_examples(args, model, args.jsonl)
    if args.valid_jsonl:
        print(f"Loading JARVIS BEC valid examples from {args.valid_jsonl}")
        train_items = examples
        valid_items, skipped_valid = load_jarvis_examples(
            args, model, args.valid_jsonl, max_structures=None
        )
        skipped.extend(skipped_valid)
    else:
        train_items, valid_items = split_examples(examples, args.valid_fraction, args.seed)
    print(
        f"usable={len(train_items) + len(valid_items)} "
        f"train={len(train_items)} valid={len(valid_items)} "
        f"skipped={len(skipped)} trainable_params={n_trainable} "
        f"devices={','.join(map(str, devices))} global_batch={args.batch_size}"
    )

    with open(outdir / "trainable_parameters.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(trainable_names))
    if init_rows:
        with open(outdir / "randomized_parameters.tsv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["name", "shape", "init_mode", "std_used"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(init_rows)
    with open(outdir / "skipped.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["jsonl", "index", "reason"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(skipped)

    before = evaluate(
        model,
        valid_items,
        args,
        device,
        write_rows_path=outdir / "valid_bec_predictions_before.tsv",
    )
    print("Before fine-tuning:", before)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_mae = before["mae"]
    best_path = outdir / "MACELES-uiu-jarvis-bec-ft-best.model"
    last_path = outdir / "MACELES-uiu-jarvis-bec-ft-last.model"
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=len(models))
        if len(models) > 1
        else None
    )
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            bec_losses = []
            asr_losses = []
            for batch_items in iter_minibatches(
                train_items, args.batch_size, shuffle=True, seed=args.seed + epoch
            ):
                optimizer.zero_grad(set_to_none=True)
                if executor is None:
                    batch, target = batch_to_device(batch_items, device)
                    pred = predict_bec(model, batch, training=True)
                    bec_loss = graphwise_bec_loss(pred, target, batch.ptr, args)
                    zero_sum_loss = asr_loss(pred, batch.ptr)
                    loss = bec_loss + args.asr_weight * zero_sum_loss
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    bec_losses.append(float(bec_loss.detach().cpu()))
                    asr_losses.append(float(zero_sum_loss.detach().cpu()))
                    del batch, target, pred, loss, bec_loss, zero_sum_loss
                else:
                    loss_value, bec_value, asr_value = synchronous_data_parallel_step(
                        models,
                        devices,
                        batch_items,
                        args,
                        optimizer,
                        trainable_names,
                        executor,
                    )
                    losses.append(loss_value)
                    bec_losses.append(bec_value)
                    asr_losses.append(asr_value)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_bec_loss": float(np.mean(bec_losses)),
                "train_asr_loss": float(np.mean(asr_losses)),
            }
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                valid_metrics = evaluate(model, valid_items, args, device)
                record.update({f"valid_{key}": value for key, value in valid_metrics.items()})
                print(
                    f"epoch {epoch:03d} train_loss={record['train_loss']:.6f} "
                    f"valid_mae={valid_metrics['mae']:.6f} "
                    f"valid_rmse={valid_metrics['rmse']:.6f} "
                    f"valid_pearson={valid_metrics['pearson']:.6f}"
                )
                if valid_metrics["mae"] < best_mae:
                    best_mae = valid_metrics["mae"]
                    torch.save(model.cpu(), best_path)
                    model.to(device)
                    evaluate(
                        model,
                        valid_items,
                        args,
                        device,
                        write_rows_path=outdir / "valid_bec_predictions_best.tsv",
                    )
            else:
                print(f"epoch {epoch:03d} train_loss={record['train_loss']:.6f}")
            history.append(record)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    torch.save(model.cpu(), last_path)
    write_history(outdir / "history.tsv", history)
    manifest = vars(args).copy()
    manifest.update(
        {
            "n_usable": len(train_items) + len(valid_items),
            "n_train": len(train_items),
            "n_valid": len(valid_items),
            "n_skipped": len(skipped),
            "n_trainable_params": n_trainable,
            "n_randomized_parameter_tensors": len(init_rows),
            "data_parallel_devices": [str(item) for item in devices],
            "global_batch_size": args.batch_size,
            "best_valid_mae": best_mae,
            "best_model": str(best_path),
            "last_model": str(last_path),
            "before_valid_metrics": before,
        }
    )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved best model: {best_path}")
    print(f"Saved last model: {last_path}")


if __name__ == "__main__":
    main()
