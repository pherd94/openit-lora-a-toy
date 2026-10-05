"""3단계: 학습용 / 동작 확인용 분할과 학습 문자열 만들기

분할
- 무작위가 아니라 문항(item_id) 단위로 뗀다. 100건이 모두 같은 계보라서 성능 수치가 아니라 동작 확인용이다.
- 확인용 문항: Q1.D10, Q1.D11 (진단·복약), Q2.D05 (가족력), Q9_1 (중강도 일수와 그 정정) = 11건
- 학습용에는 모든 사례 종류가 남는다.

학습 문자열
- Qwen3 채팅 템플릿을 생각 모드 끔(enable_thinking=False)으로 미리 적용해 prompt / completion 문자열로 저장한다.
- completion 은 정답 + <|im_end|> 로 끝난다. TRL 이 eos 를 또 붙이지 않도록 eos 로 정확히 끝나게 한다.
- prompt 토큰이 prompt+completion 토큰의 앞부분과 똑같은지 확인한다 (경계에서 토큰이 합쳐지면 손실 범위가 어긋남).

실행: .venv/bin/python scripts/prepare_data.py
결과: prepared/{A,B}_{train,val}.jsonl, results/03_split.json
"""
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "labeled_questionnaire_100_handoff" / "data"
PREP = ROOT / "prepared"
OUT = ROOT / "results"
MODEL = "Qwen/Qwen3-0.6B"
VAL_ITEMS = {"Q1.D10", "Q1.D11", "Q2.D05", "Q9_1"}
SPLIT_NOTE = "동작 확인용 (같은 계보에서 문항 단위로 분리, 성능 수치로 쓰지 않음)"


def load(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token
    assert eos == "<|im_end|>", eos

    def render(r):
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False, enable_thinking=False)
        completion = full[len(prompt):].rstrip("\n")
        assert full.startswith(prompt) and completion.endswith(eos), r["id"]
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(prompt + completion, add_special_tokens=False)["input_ids"]
        assert f_ids[: len(p_ids)] == p_ids, f"{r['id']}: prompt/completion 경계에서 토큰이 어긋남"
        return prompt, completion

    PREP.mkdir(exist_ok=True)
    summary = {"val_items": sorted(VAL_ITEMS), "note": SPLIT_NOTE}
    for name, fname in [("A", "extraction.jsonl"), ("B", "response.jsonl")]:
        splits = {"train": [], "val": []}
        for r in load(fname):
            prompt, completion = render(r)
            split = "val" if r["item_id"] in VAL_ITEMS else "train"
            splits[split].append({
                "id": r["id"],
                "category": r["category"],
                "item_id": r["item_id"],
                "split_note": SPLIT_NOTE if split == "val" else "학습용",
                "prompt": prompt,          # 템플릿 적용된 문자열 (학습용)
                "completion": completion,  # 정답 + <|im_end|>
                "messages": r["prompt"],   # 원래 메시지 (평가·서빙 확인용)
                "gold": r["completion"][0]["content"],
            })
        for split, rows in splits.items():
            with open(PREP / f"{name}_{split}.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[name] = {
            split: {"rows": len(rows), "categories": dict(Counter(r["category"] for r in rows).most_common())}
            for split, rows in splits.items()
        }

    train_cats = set(summary["A"]["train"]["categories"])
    all_cats = {r["category"] for r in load("extraction.jsonl")}
    summary["categories_missing_from_train"] = sorted(all_cats - train_cats)
    OUT.mkdir(exist_ok=True)
    (OUT / "03_split.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
