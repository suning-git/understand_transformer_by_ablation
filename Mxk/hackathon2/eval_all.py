import os, sys, torch

os.environ["NANOINFRA_BASE_DIR"] = "/root/autodl-tmp/nanoinfra/outputs"
sys.path.insert(0, "/root/autodl-tmp/nanoinfra")

torch.cuda.set_device(0)

from core.model.gpt import GPT
from core.training.model_setup import load_system
from modalities.text import TextEvaluator
from modalities.text.tokenizer import get_tokenizer

CKPTS = [
    ("Round 3b (d12 bs=8 lr6e-4)", "/root/autodl-tmp/nanoinfra/models/hackathon2/run3_2x5090_d12/step_003599", 8),
    ("Round 4 (d12 bs=16 lr6e-4)", "/root/autodl-tmp/nanoinfra/models/hackathon2/run4_d12_bs16/step_005099", 16),
    ("Round 5 (d12 bs=8 lr1e-3)", "/root/autodl-tmp/nanoinfra/models/hackathon2/run5_lr1e3/step_003599", 8),
]

for name, ckpt_dir, bs in CKPTS:
    setup = load_system(ckpt_dir, trunk_cls=GPT, sequence_len=2048, use_compile=False)
    system = setup["system"]
    device = setup["device"]
    param_count = sum(p.numel() for p in system.parameters()) / 1e9
    meta = setup["meta"]
    st = meta["trainer_state"]
    print("{}: {:.2f}B params, step={}, train_loss={:.4f}, time={:.1f}min".format(
        name, param_count, meta["step"], st["smooth_train_loss"], st["total_training_time"] / 60))

    eval_config = {"interval_steps": 300, "eval_tokens": 1048576}
    evaluator = TextEvaluator(eval_config, device_batch_size=bs, sequence_len=2048)
    result = evaluator.evaluate(system, torch.amp.autocast("cuda", dtype=torch.bfloat16))
    print("  val CE: {:.4f}, val BPB: {:.4f}\n".format(result["val/text_ce"], result["val/bpb"]))
