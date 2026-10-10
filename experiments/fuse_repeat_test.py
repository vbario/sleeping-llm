"""Requantization check — do small LoRA deltas survive repeated 4-bit fuses?

`mlx_lm.fuse` on a 4-bit model dequantizes each LoRA-wrapped linear, adds
scale * B @ A, and re-quantizes to 4 bits (mlx_lm/tuner/lora.py, LoRALinear.fuse).
Every sleep cycle therefore rounds the whole weight matrix again. This script
measures two things:

  1. delta survival (weight level, no generation noise): for every LoRA-touched
     matrix, ||dequant(W_fused) - dequant(W_base) - delta|| / ||delta||. 0 means
     the delta landed exactly; 1 means it was rounded away entirely.
  2. erosion over cycles: train a tiny adapter on ONE probe fact, fuse it, then
     run K more train+fuse cycles on unrelated filler facts (no replay) and
     track held-out recall of the probe fact and perplexity after every fuse.

MLX only (the question is specific to mlx_lm's fuse). Needs roughly one
fused model copy of disk space (~1.8 GB for the 3B 4-bit model) since each
fused model is deleted before the next is written.

    python3 experiments/fuse_repeat_test.py --cycles 6
    python3 experiments/fuse_repeat_test.py --cycles 3 --iters 30 --out /tmp/fuse_repeat.json
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config  # noqa: E402
from src.memory.facts import QAPair  # noqa: E402
from src.sleep.paraphrase import ParaphraseGenerator  # noqa: E402
from src.sleep.probe import RecallProbe  # noqa: E402
from src.sleep.trainer import SleepTrainer  # noqa: E402

PROBE = QAPair(question="What is the user's dog's name?",
               answer="The user's dog is named Pebble.", value="Pebble", priority=1.0)
FILLERS = [
    QAPair(question="What is the user's favorite season?", answer="The user's favorite season is autumn.", value="autumn"),
    QAPair(question="What instrument does the user play?", answer="The user plays the cello.", value="cello"),
    QAPair(question="Where did the user study?", answer="The user studied at Uppsala University.", value="Uppsala University"),
    QAPair(question="What is the user's usual coffee order?", answer="The user's usual coffee order is a cortado.", value="cortado"),
    QAPair(question="What language does the user speak besides English?", answer="The user speaks Norwegian besides English.", value="Norwegian"),
    QAPair(question="What is the user's favorite book?", answer="The user's favorite book is Piranesi.", value="Piranesi"),
    QAPair(question="What car does the user drive?", answer="The user drives a yellow Fiat Panda.", value="yellow Fiat Panda"),
    QAPair(question="What is the user allergic to?", answer="The user is allergic to sesame.", value="sesame"),
]
REF_TEXT = ("The town library opens at nine and closes at six on weekdays. Its reading room faces a "
            "small courtyard where a plane tree has grown for over a century. Visitors may borrow up "
            "to ten books at a time, and the catalogue can be searched online.")


def delta_survival(base_path, fused_path, adapter_path):
    """Per-matrix relative error between the applied delta and the delta the fuse kept."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx.utils import tree_flatten

    adapters = mx.load(str(Path(adapter_path) / "adapters.safetensors"))
    cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
    lp = cfg.get("lora_parameters", {})
    scale = float(lp.get("scale", 20.0))

    base_model, _ = load(base_path)
    fused_model, _ = load(str(fused_path))
    base_params = dict(tree_flatten(base_model.parameters()))
    fused_params = dict(tree_flatten(fused_model.parameters()))

    def dequant(params, prefix):
        w = params[prefix + ".weight"]
        if prefix + ".scales" in params:
            mod = base_model
            for part in prefix.split("."):
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
            return mx.dequantize(w, params[prefix + ".scales"], params[prefix + ".biases"],
                                 group_size=mod.group_size, bits=mod.bits)
        return w

    rows = []
    for key in adapters:
        if not key.endswith(".lora_a"):
            continue
        prefix = key[:-len(".lora_a")]
        lora_a = adapters[key]
        lora_b = adapters[prefix + ".lora_b"]
        delta = ((scale * lora_b.T) @ lora_a.T).astype(mx.float32)
        w_base = dequant(base_params, prefix).astype(mx.float32)
        w_fused = dequant(fused_params, prefix).astype(mx.float32)
        observed = w_fused - w_base
        err = mx.linalg.norm(observed - delta).item()
        ref = mx.linalg.norm(delta).item()
        cos = (mx.sum(observed * delta) / (mx.linalg.norm(observed) * mx.linalg.norm(delta) + 1e-12)).item()
        rows.append({"matrix": prefix, "delta_norm": ref, "rel_error": err / ref if ref else None,
                     "cosine": cos, "weight_norm": mx.linalg.norm(w_base).item()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(project_root / "config.yaml"))
    ap.add_argument("--cycles", type=int, default=6, help="filler train+fuse cycles after the probe fuse")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", default=str(project_root / "experiments" / "results" / "fuse_repeat.json"))
    ap.add_argument("--workdir", default=str(project_root / "experiments" / "results" / "fuse_repeat_work"))
    args = ap.parse_args()

    config = Config(args.config)
    if config.model.get("backend", "mlx") != "mlx":
        sys.exit("This check is specific to mlx_lm.fuse; set model.backend: mlx")
    from src.backend.mlx_backend import MLXBackend
    backend = MLXBackend(config)
    backend.load()
    base_path = config.model["path"]
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    trainer = SleepTrainer(config, backend)
    gen = ParaphraseGenerator(config, backend)
    probe = RecallProbe(backend, config.context.get("system_prompt", ""))
    train, held_out, variants = gen.generate_for(PROBE)
    PROBE.paraphrases, PROBE.held_out, PROBE.answer_variants = train, held_out, variants

    results = {"base_model": base_path, "cycles": [], "iters": args.iters, "started": time.strftime("%c")}
    ppl0 = backend.compute_perplexity(REF_TEXT)
    r0 = probe.test(PROBE, [])
    results["before"] = {"ppl": ppl0, "probe_recall": r0["fraction"]}
    print(f"base: ppl={ppl0:.3f} probe_recall={r0['fraction']}")

    def cycle(name, facts):
        fused = trainer.train_and_fuse(facts, name, work, weighted=True, iters=args.iters)
        if fused is None:
            sys.exit(f"cycle {name}: train/fuse failed")
        return fused

    # Cycle 0: the probe fact
    fused = cycle("probe", [PROBE])
    survival = delta_survival(backend.model_path, fused, work / "adapters" / "probe")
    backend.reload(fused)
    ppl = backend.compute_perplexity(REF_TEXT)
    rec = probe.test(PROBE, [])
    mean_err = sum(r["rel_error"] for r in survival if r["rel_error"] is not None) / max(1, len(survival))
    results["cycles"].append({"cycle": 0, "facts": [PROBE.value], "ppl": ppl, "probe_recall": rec["fraction"],
                              "delta_rel_error_mean": mean_err, "delta_survival": survival})
    print(f"cycle 0 (probe fused): ppl={ppl:.3f} probe_recall={rec['fraction']} "
          f"mean delta rel. error={mean_err:.3f}")
    previous = fused

    # Cycles 1..K: filler facts, no replay — pure erosion of the probe fact
    for k in range(1, args.cycles + 1):
        filler = FILLERS[(k - 1) % len(FILLERS)]
        ft, fh, fv = gen.generate_for(filler)
        filler.paraphrases, filler.held_out, filler.answer_variants = ft, fh, fv
        fused = cycle(f"filler{k:02d}", [filler])
        survival = delta_survival(previous, fused, work / "adapters" / f"filler{k:02d}")
        backend.reload(fused)
        shutil.rmtree(previous, ignore_errors=True)        # keep disk use to one copy
        ppl = backend.compute_perplexity(REF_TEXT)
        rec = probe.test(PROBE, [])
        frec = probe.test(filler, [])
        mean_err = sum(r["rel_error"] for r in survival if r["rel_error"] is not None) / max(1, len(survival))
        results["cycles"].append({"cycle": k, "facts": [filler.value], "ppl": ppl,
                                  "probe_recall": rec["fraction"], "filler_recall": frec["fraction"],
                                  "delta_rel_error_mean": mean_err})
        print(f"cycle {k}: ppl={ppl:.3f} probe_recall={rec['fraction']} filler_recall={frec['fraction']} "
              f"mean delta rel. error={mean_err:.3f}")
        previous = fused
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
