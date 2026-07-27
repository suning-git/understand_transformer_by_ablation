import os, sys, torch

os.environ["NANOINFRA_BASE_DIR"] = "/root/autodl-tmp/nanoinfra/outputs"
sys.path.insert(0, "/root/autodl-tmp/nanoinfra")

torch.cuda.set_device(0)

from core.model.gpt import GPT
from core.training.model_setup import load_system
from modalities.text import TextEvaluator
from modalities.text.tokenizer import get_tokenizer

ckpt_dir = "/root/autodl-tmp/nanoinfra/models/hackathon2/run4_d12_bs16/step_005099"

print("Loading checkpoint...")
setup = load_system(ckpt_dir, trunk_cls=GPT, sequence_len=2048, use_compile=False)
system = setup["system"]
device = setup["device"]

param_count = sum(p.numel() for p in system.parameters()) / 1e9
print(f"Model: {system.arch}, {param_count:.2f}B params")
print(f"Step: {setup['meta']['step']}")
print(f"Training loss: {setup['meta']['trainer_state']['smooth_train_loss']:.4f}")

eval_config = {"interval_steps": 300, "eval_tokens": 1048576}
print("Running eval...")
evaluator = TextEvaluator(eval_config, device_batch_size=16, sequence_len=2048)
result = evaluator.evaluate(system, torch.amp.autocast("cuda", dtype=torch.bfloat16))
print("val CE: {:.4f}, val BPB: {:.4f}".format(result["val/text_ce"], result["val/bpb"]))
