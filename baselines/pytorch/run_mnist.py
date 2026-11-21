"""Team CLI around the preserved author models and compiled attention library."""
import argparse
import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from experiment import (DEFAULT_CONFIG, build, checkpoint_meta, configure, environment,
                        evaluate, generate, load_checkpoint, load_mnist, load_model,
                        mol_nll, save_grid, sync, verify_sources, write_json)
from source_check import ROOT, sha256


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "benchmark"):
        s = sub.add_parser(name)
        s.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        s.add_argument("--threads", type=int)
        s.add_argument("--seed", type=int, default=1229)
        s.add_argument("--attention", choices=["causal-linear", "full"])
        if name in ("train", "evaluate"):
            s.add_argument("--data-dir", required=True)
            s.add_argument("--batch-size", type=int, default=10)
            s.add_argument("--max-batches", type=int, default=0, help="0=all; nonzero is debugging only")
        if name == "train":
            s.add_argument("--epochs", type=int, default=1, help="Total target epochs, including resumed epochs")
            s.add_argument("--lr", type=float, default=1e-4)
            s.add_argument("--run-dir", required=True)
            s.add_argument("--resume", help="New-format last.pt with optimizer and RNG")
        else:
            s.add_argument("--weights", required=True)
            s.add_argument("--out", required=True)
        if name == "benchmark":
            s.add_argument("--modes", choices=["recurrent", "parallel", "both"], default="recurrent")
            s.add_argument("--batch-sizes", default="1,8,32")
            s.add_argument("--warmup", type=int, default=3)
            s.add_argument("--repeats", type=int, default=10)
            s.add_argument("--allocator-cache", choices=["warm", "cold"], default="warm",
                           help="warm: retain allocator cache after warmup; cold: legacy empty_cache before each trial (not model KV cache)")
            s.add_argument("--samples", help="Optional PNG outside timing; uses first measured mode")
            s.add_argument("--sample-count", type=int, default=8)
    return p


def metadata(args, device):
    return {"command": sys.argv[1:], "arguments": vars(args), "environment": environment(device),
            "author_sources": verify_sources(),
            "harness_sha256": {name: sha256(ROOT / name) for name in ("run_mnist.py", "experiment.py", "author_model.py")}}


def train(args):
    resume = None
    if args.resume:
        state, config, resume = load_checkpoint(args.resume, args.attention)
        if resume["format"] != "author-mnist-v1":
            raise ValueError("Author raw weights have no optimizer/RNG; use them for evaluation or start a new training run")
        explicit = {x.split("=")[0] for x in sys.argv[1:] if x.startswith("--")}
        for key in ("batch_size", "lr", "seed", "max_batches"):
            if "--" + key.replace("_", "-") in explicit and getattr(args, key) != resume["training"][key]:
                raise ValueError("Resume setting conflicts with checkpoint: " + key)
            setattr(args, key, resume["training"][key])
    else:
        if not args.attention:
            raise ValueError("New training requires --attention")
        config = dict(DEFAULT_CONFIG, attention_type=args.attention)
    if args.epochs < 1 or args.batch_size < 1 or args.max_batches < 0:
        raise ValueError("Invalid training budget")
    device = configure(args.device, args.seed, args.threads)
    run = Path(args.run_dir)
    if (run / "last.pt").exists() and (not args.resume or Path(args.resume).resolve() != (run / "last.pt").resolve()):
        raise FileExistsError("Run already exists; explicitly resume its last.pt or use a new directory")
    train_x, _, data_id = load_mnist(args.data_dir)
    net = build(config).to(device)
    optimizer = torch.optim.RAdam(net.parameters(), lr=args.lr)
    start, history = 1, []
    if resume:
        if resume["data"]["files"] != data_id["files"]:
            raise ValueError("Resume dataset changed")
        net.load_state_dict(state, strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start, history = resume["epoch"] + 1, list(resume["history"])
        torch.set_rng_state(resume["rng_cpu"].cpu())
        if resume["rng_cuda"] and device.type == "cuda":
            if len(resume["rng_cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count changed on resume")
            torch.cuda.set_rng_state_all([s.cpu() for s in resume["rng_cuda"]])
    if start > args.epochs:
        raise ValueError("Target epochs must exceed checkpoint epoch")
    run.mkdir(parents=True, exist_ok=True)
    training = {k: getattr(args, k) for k in ("batch_size", "lr", "seed", "max_batches")}
    info = dict(metadata(args, device), data=data_id, config=config, selection="fixed_budget_last_epoch")
    write_json(run / ("resume_%d.json" % start if resume else "run.json"), info)
    for epoch in range(start, args.epochs + 1):
        net.train()
        total, seen, steps = 0.0, 0, 0
        order = torch.randperm(len(train_x))
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        for offset in range(0, len(train_x), args.batch_size):
            if args.max_batches and steps >= args.max_batches:
                break
            pixels = train_x[order[offset:offset + args.batch_size]].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            nll = mol_nll(net(pixels[:, :-1]), pixels[:, 1:])
            loss = nll.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(nll.detach().sum())
            seen += len(pixels)
            steps += 1
            if steps % 100 == 0:
                print("epoch=%d step=%d bpd=%.4f" % (epoch, steps, float(loss.detach()) / math.log(2)), flush=True)
        sync(device)
        row = {"epoch": epoch, "train_bpd_predicted_783": total / (seen * 783 * math.log(2)),
               "samples": seen, "steps": steps, "train_seconds": time.perf_counter() - begin,
               "full_epoch": seen == len(train_x),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        history.append(row)
        checkpoint = {"format": "author-mnist-v1", "state_dict": net.state_dict(), "config": config,
                      "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history,
                      "training": training, "data": data_id,
                      "author_sources": info["author_sources"], "rng_cpu": torch.get_rng_state(),
                      "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        torch.save(checkpoint, run / "last.tmp")
        (run / "last.tmp").replace(run / "last.pt")
        write_json(run / "history.json", history)
        print(json.dumps(row), flush=True)
    return history


def eval_weights(args):
    if args.batch_size < 1 or args.max_batches < 0:
        raise ValueError("Invalid evaluation batch settings")
    device = configure(args.device, args.seed, args.threads)
    net, config, info = load_model(args.weights, args.attention, device)
    _, test_x, identity = load_mnist(args.data_dir)
    result = dict(metadata(args, device), data=identity, checkpoint=checkpoint_meta(args.weights, config, info),
                  metrics=evaluate(net, test_x, device, args.batch_size, args.max_batches))
    write_json(args.out, result)
    print(json.dumps(result["metrics"], indent=2))
    return result


def clear(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def measure(net, batch, mode, config, args, device):
    # Configuration-level cleanup precedes warmup. Never release warmed allocator
    # blocks between steady-state trials. Model KV/state is still reset by generate.
    clear(device)
    for repeat in range(args.warmup):
        torch.manual_seed(args.seed + 10000 + repeat)
        generate(net, batch, device, mode, config["sequence_length"])
    records = []
    for repeat in range(args.repeats):
        if args.allocator_cache == "cold":
            clear(device)
        torch.manual_seed(args.seed + repeat)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = datetime.now(timezone.utc).isoformat()
        begin = time.perf_counter()
        pixels = generate(net, batch, device, mode, config["sequence_length"])
        sync(device)
        spent = time.perf_counter() - begin
        finished = datetime.now(timezone.utc).isoformat()
        records.append({"repeat": repeat, "seed": args.seed + repeat,
                        "started_utc": started, "finished_utc": finished,
                        "seconds": spent, "images_per_second": batch / spent,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        # Validation and host transfers are outside the timed section.
        if pixels.shape != (batch, config["sequence_length"] + 1) or pixels.dtype != torch.int64:
            raise RuntimeError("Generation returned an unexpected shape or dtype")
        if repeat == 0 and (pixels.min().item() < 0 or pixels.max().item() > 255):
            raise RuntimeError("Generation returned an invalid MNIST pixel")
        del pixels
    rates = [r["images_per_second"] for r in records]
    return {"status": "ok", "requested_batch": batch, "actual_batch": batch, "mode": mode,
            "median_images_per_second": statistics.median(rates), "stdev_images_per_second": statistics.stdev(rates),
            "median_batch_seconds": statistics.median(r["seconds"] for r in records),
            "min_images_per_second": min(rates), "max_images_per_second": max(rates), "trials": records}


def benchmark(args):
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    if min(sizes) < 1 or args.repeats < 3 or args.warmup < 1 or args.sample_count < 1:
        raise ValueError("Positive batches/sample-count, >=3 repeats, >=1 warmup required")
    device = configure(args.device, args.seed, args.threads)
    result = dict(metadata(args, device), rows=[], protocol={"batch_backoff": False,
                  "version": "mnist-generation-v2", "allocator_cache": args.allocator_cache,
                  "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
                  "timer": "full configured image generation with synchronization; weight loading and PNG excluded",
                  "timed_work": "generation, sampling, output assembly; excludes output checks and host copies",
                  "model_state": "fresh KV/recurrent state for every generated batch",
                  "fixed_initial_pixel": 0,
                  "memory": "PyTorch allocator peaks, not total driver/process/vGPU memory"})
    modes = ["recurrent", "parallel"] if args.modes == "both" else [args.modes]
    for mode in modes:
        net, config, info = load_model(args.weights, args.attention, device, recurrent=mode == "recurrent")
        result["checkpoint"] = checkpoint_meta(args.weights, config, info)
        result["protocol"].update(sequence_length=config["sequence_length"], output_pixels=config["sequence_length"] + 1)
        for size in sizes:
            print("attention=%s mode=%s batch=%d" % (config["attention_type"], mode, size), flush=True)
            try:
                row = measure(net, size, mode, config, args, device)
            except RuntimeError as error:
                if not isinstance(error, torch.OutOfMemoryError) and "out of memory" not in str(error).lower():
                    raise
                row = {"status": "oom", "requested_batch": size, "actual_batch": None, "mode": mode, "error": str(error)}
            clear(device)
            result["rows"].append(row)
            write_json(args.out, result)
            print(json.dumps({k: v for k, v in row.items() if k != "trials"}), flush=True)
        if args.samples and mode == modes[0]:
            torch.manual_seed(args.seed)
            save_grid(generate(net, args.sample_count, device, mode, config["sequence_length"]), args.samples)
        del net
        clear(device)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    verify_sources()
    if args.command != "train" and Path(args.out).exists():
        raise FileExistsError("Use a new --out to preserve previous results")
    return {"train": train, "evaluate": eval_weights, "benchmark": benchmark}[args.command](args)


if __name__ == "__main__":
    main()
