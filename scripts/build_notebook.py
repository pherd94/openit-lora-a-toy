"""4단계: LoRA-A 토이 학습용 Colab 노트북 만들기

실행: python3 scripts/build_notebook.py
결과: notebooks/lora_a_toy_qwen3_0.6b.ipynb, colab_upload_A.zip (Colab 에 올릴 파일 묶음)
"""
import base64
import io
import json
import lzma
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# LoRA-A 토이 학습 (Qwen3-0.6B)

사용자 응답 → Facts · Status · Evidence JSON 추출 어댑터를 작게 학습해 보는 노트북입니다.
목적은 **학습이 돌아가고 JSON이 나오는지 확인**하는 것입니다. 100건이 모두 같은 계보라서 여기 나오는 숫자를 성능 수치로 쓰지 않습니다.

순서
1. 런타임 → 런타임 유형 변경 → GPU 선택 (T4 이상)
2. 아래 셀을 위에서부터 차례로 실행
3. 업로드 셀에서 `colab_upload_A.zip` 선택
4. 마지막 셀에서 결과 묶음(`lora_a_result.zip`)을 내려받기

설정: rank 16 / alpha 16, q·k·v·o·gate·up·down 7종, 학습률 2e-4, 배치 1 × 누적 8, 정답 부분만 손실, 생각 모드 끔.
""")

code("""
# 1. 설치 (Colab 기본 torch 는 그대로 쓰고, 학습 도구만 최신으로)
!pip install -q -U transformers trl peft accelerate datasets
!pip install -q liger-kernel || echo "liger-kernel 설치 실패: 메모리 절약 옵션 없이 진행"
!pip uninstall -y -q torchao   # Colab 에 깔린 옛 torchao(0.10)가 최신 peft 와 충돌. 이번 학습에는 쓰지 않음
""")

code("""
# 2. 설정값 (여기만 바꾸면 됩니다)
MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 12288        # 실측 최대 10,746 토큰 + 여유. 이보다 짧으면 정답이 잘려 나감
EPOCHS = 3                # 작업 노트 범위 1~3
LR = 2e-4
GRAD_ACCUM = 8
LORA_R, LORA_ALPHA = 16, 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_NEW_TOKENS = 400      # 정답은 최대 190 토큰
SEED = 42
SMOKE = False             # True 면 2건으로 1 step 만 돌려 코드만 확인
""")

code("""
# 3. 데이터 올리기: colab_upload_A.zip (A_train.jsonl, A_val.jsonl, eval_a.py)
import os, zipfile
if not os.path.exists("A_train.jsonl"):
    from google.colab import files
    uploaded = files.upload()
    zipfile.ZipFile(next(iter(uploaded))).extractall(".")
print(sorted(f for f in os.listdir(".") if f.endswith((".jsonl", ".py"))))
""")

code("""
# 4. 환경 확인
import json, time, platform, importlib
import torch, transformers, trl, peft

has_cuda = torch.cuda.is_available()
gpu = torch.cuda.get_device_name(0) if has_cuda else "none"
# T4(7.5 세대)는 bf16 을 흉내만 내므로 is_bf16_supported() 대신 세대로 판단 (A100·L4 등 8 이상만 bf16)
use_bf16 = has_cuda and torch.cuda.get_device_capability(0)[0] >= 8
try:
    importlib.import_module("liger_kernel")
    use_liger = has_cuda
except ImportError:
    use_liger = False
versions = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "trl": trl.__version__, "peft": peft.__version__}
total_mem = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if has_cuda else None
print(f"GPU: {gpu} ({total_mem} GB) | bf16: {use_bf16} | liger: {use_liger}")
print(versions)
if has_cuda and not use_liger:
    print("주의: liger 가 없으면 11,000 토큰 출력 계산에 수 GB 가 더 듭니다. T4 에서는 메모리가 부족할 수 있습니다.")
""")

code("""
# 5. 데이터 읽기
from datasets import Dataset

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_rows, val_rows = load_jsonl("A_train.jsonl"), load_jsonl("A_val.jsonl")
if SMOKE:
    train_rows, val_rows = train_rows[:2], val_rows[:1]
train_ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in train_rows])
val_ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in val_rows])
print(f"학습 {len(train_ds)}건, 확인용 {len(val_ds)}건 ({val_rows[0]['split_note']})")
print(train_rows[0]["prompt"][-80:])
print(train_rows[0]["completion"][:120])
""")

code("""
# 6. 모델과 LoRA 설정
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig

set_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
assert tokenizer.eos_token == "<|im_end|>"
# T4 처럼 bf16 이 없으면 원본을 fp32 로 올리고 fp16 혼합정밀로 학습 (LoRA 가중치가 fp16 이면 학습이 깨짐)
load_dtype = torch.bfloat16 if use_bf16 else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=load_dtype, attn_implementation="sdpa",
                                             device_map={"": 0} if has_cuda else None)
peft_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, target_modules=TARGET_MODULES,
                         bias="none", task_type="CAUSAL_LM")

# Qwen3 는 k·v 머리 수(8)가 q(16)보다 적다. transformers 가 enable_gqa=True 로 어텐션을 부르면
# 메모리 절약 커널이 "머리 수가 같아야 함" 조건에 걸려 전체 행렬 경로로 빠지고, 1만 토큰에서 T4 메모리가 터진다.
# k·v 를 복제해 넘기도록 바꾼다 (계산 결과는 같음).
import transformers.integrations.sdpa_attention as sdpa_attention
sdpa_attention.use_gqa_in_sdpa = lambda *a, **kw: False
""")

code("""
# 7. 학습 설정
from trl import SFTConfig, SFTTrainer

args = SFTConfig(
    output_dir="lora_a_out",
    max_length=MAX_LENGTH,
    completion_only_loss=True,        # 정답 부분만 손실
    packing=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1 if SMOKE else GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    max_steps=1 if SMOKE else -1,
    learning_rate=LR,
    lr_scheduler_type="linear",
    warmup_steps=0.1,                 # 소수면 전체 step 대비 비율 (transformers 5)
    bf16=use_bf16,
    fp16=has_cuda and not use_bf16,
    use_liger_kernel=use_liger,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},   # False 는 fp16 autocast 와 저장 텐서 개수 대조에서 충돌
    eval_strategy="epoch",
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)
trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                     processing_class=tokenizer, peft_config=peft_config)
trainer.model.print_trainable_parameters()

# 손실이 정답 부분에만 걸리는지, 잘린 행이 없는지 확인
def n_loss_tokens(ex):
    # TRL 버전에 따라 completion_mask 를 주거나, 프롬프트를 -100 으로 가린 labels 를 준다
    if "completion_mask" in ex:
        return sum(ex["completion_mask"])
    return sum(1 for x in ex["labels"] if x != -100)

n_completion = [len(tokenizer(r["completion"], add_special_tokens=False)["input_ids"]) for r in train_rows]
n_loss = [n_loss_tokens(ex) for ex in trainer.train_dataset]
print("손실이 걸리는 토큰 수 (앞 3건):", n_loss[:3], "| 정답 토큰 수:", n_completion[:3])
lengths = [len(x["input_ids"]) for x in trainer.train_dataset]
print("학습 행 길이 최대:", max(lengths), "| MAX_LENGTH 에 닿은 행:", sum(l >= MAX_LENGTH for l in lengths))
assert n_loss == n_completion, "손실 범위가 정답 부분과 다릅니다"
""")

code("""
# 8. 학습
if has_cuda:
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
train_out = trainer.train()
train_time_s = round(time.time() - t0, 1)
peak_mem_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if has_cuda else None
print(f"학습 시간 {train_time_s}s, 최대 메모리 {peak_mem_gb} GB")
""")

code("""
# 9. 학습 곡선
import matplotlib.pyplot as plt

log = trainer.state.log_history
tr = [(x["epoch"], x["loss"]) for x in log if "loss" in x]
ev = [(x["epoch"], x["eval_loss"]) for x in log if "eval_loss" in x]
plt.figure(figsize=(6, 3.5))
plt.plot(*zip(*tr), label="train loss", marker=".")
if ev:
    plt.plot(*zip(*ev), label="val loss (check only)", marker="o")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150); plt.show()
""")

code("""
# 10. 어댑터 저장 (vLLM 에 그대로 붙일 수 있는 형식)
ADAPTER_DIR = "lora_a_qwen3_0.6b_toy"
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(os.listdir(ADAPTER_DIR))
""")

code("""
# 11. 생성: 확인용 전체 + 학습 데이터 사례 종류별 1건, 그리고 어댑터 끈 원본 모델 비교
model = trainer.model
model.eval()
model.config.use_cache = True

def sync():
    if has_cuda:
        torch.cuda.synchronize()

@torch.inference_mode()
def run(row):
    enc = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
    sync(); t0 = time.time()
    model.generate(**enc, max_new_tokens=1, do_sample=False)       # 첫 토큰까지 시간
    sync(); ttft = time.time() - t0
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    sync(); total = time.time() - t0
    new = out[0, enc["input_ids"].shape[1]:]
    return {"id": row["id"], "output": tokenizer.decode(new, skip_special_tokens=True),
            "ttft_s": round(ttft, 3), "total_s": round(total, 3), "new_tokens": int(new.shape[0]),
            "prompt_tokens": int(enc["input_ids"].shape[1])}

run(val_rows[0])  # 준비 호출: 첫 호출의 초기화 시간이 측정에 섞이지 않게

seen, train_probe = set(), []
for r in train_rows:
    if r["category"] not in seen:
        seen.add(r["category"]); train_probe.append(r)

preds = {"val_lora": [run(r) for r in val_rows], "train_lora": [run(r) for r in train_probe]}
with model.disable_adapter():
    preds["val_base"] = [run(r) for r in val_rows]
for name, rows in preds.items():
    with open(f"pred_{name}.jsonl", "w", encoding="utf-8") as f:
        for p in rows:
            f.write(json.dumps(p, ensure_ascii=False) + "\\n")
print(preds["val_lora"][0]["output"][:500])
""")

code("""
# 12. 평가
from eval_a import evaluate

golds = {"val_lora": val_rows, "train_lora": train_probe, "val_base": val_rows}
metrics = {name: evaluate(preds[name], golds[name]) for name in preds}
keys = ["json_parse_rate", "top_keys_ok_rate", "exact_row_match_rate", "fact_found_rate", "fact_value_match_rate",
        "fact_status_match_rate", "intent_match_rate", "relation_match_rate",
        "evidence_text_in_source_rate", "evidence_offset_correct_rate", "ttft_s_median", "total_s_median"]
print(f"{'항목':34s}" + "".join(f"{n:>12s}" for n in metrics))
for k in keys:
    print(f"{k:34s}" + "".join(f"{str(m['metrics'].get(k)):>12s}" for m in metrics.values()))
""")

code("""
# 13. 결과 묶어서 내려받기
import zipfile  # 데이터 내장 버전은 3번 셀에서 zipfile 을 불러오지 않으므로 여기서 다시
run_info = {"gpu": gpu, "gpu_total_gb": total_mem, "bf16": use_bf16, "liger": use_liger, "versions": versions,
            "train_time_s": train_time_s, "peak_mem_gb": peak_mem_gb, "smoke": SMOKE,
            "config": {"model": MODEL_ID, "max_length": MAX_LENGTH, "epochs": EPOCHS, "lr": LR, "grad_accum": GRAD_ACCUM,
                       "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "target_modules": TARGET_MODULES, "seed": SEED},
            "rows": {"train": len(train_rows), "val": len(val_rows)},
            "split_note": val_rows[0]["split_note"]}
json.dump(run_info, open("run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(trainer.state.log_history, open("log_history.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(metrics, open("metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

with zipfile.ZipFile("lora_a_result.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["run_info.json", "log_history.json", "metrics.json", "loss_curve.png"] + [f"pred_{n}.jsonl" for n in preds]:
        z.write(f)
    for f in os.listdir(ADAPTER_DIR):
        z.write(os.path.join(ADAPTER_DIR, f))
try:
    from google.colab import files
    files.download("lora_a_result.zip")
except ImportError:
    print("lora_a_result.zip 저장됨")
""")

nb = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
(ROOT / "notebooks").mkdir(exist_ok=True)
out = ROOT / "notebooks" / "lora_a_toy_qwen3_0.6b.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

upload_files = [ROOT / "prepared" / "A_train.jsonl", ROOT / "prepared" / "A_val.jsonl", ROOT / "scripts" / "eval_a.py"]
with zipfile.ZipFile(ROOT / "colab_upload_A.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in upload_files:
        z.write(p, p.name)

# 데이터까지 셀 안에 넣은 한 파일짜리 노트북 (업로드 단계 없음). 프롬프트 반복이 많아 xz 로 약 14KB.
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as t:
    for p in upload_files:
        t.add(p, arcname=p.name)
data_b64 = base64.b64encode(lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME)).decode()
embedded = f'''# 3. 데이터 풀기: A_train.jsonl, A_val.jsonl, eval_a.py 가 이 셀 안에 압축되어 있음
import base64, io, lzma, os, tarfile
DATA_B64 = "{data_b64}"
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(base64.b64decode(DATA_B64)))) as t:
    t.extractall(".")
print(sorted(f for f in os.listdir(".") if f.endswith((".jsonl", ".py"))))
'''
nb_embedded = json.loads(json.dumps(nb))
for c in nb_embedded["cells"]:
    if c["cell_type"] == "code" and c["source"][0].startswith("# 3."):
        c["source"] = embedded.splitlines(keepends=True)
    if c["cell_type"] == "markdown":
        c["source"] = [s.replace("3. 업로드 셀에서 `colab_upload_A.zip` 선택", "3. 데이터는 3번 셀 안에 들어 있어 업로드할 파일 없음") for s in c["source"]]
out2 = ROOT / "notebooks" / "lora_a_toy_qwen3_0.6b_selfcontained.ipynb"
out2.write_text(json.dumps(nb_embedded, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, out2, ROOT / "colab_upload_A.zip")
