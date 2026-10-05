"""2단계: Qwen3 토크나이저로 A, B 각 100건의 토큰 길이 실측

- 채팅 템플릿을 실제로 적용한 길이를 잰다 (생각 모드 끔: enable_thinking=False)
- prompt: system + user 에 생성 시작 표시까지 붙인 길이 (추론 때 모델이 읽는 길이)
- completion: 정답 부분 길이 (학습 손실이 걸리는 부분)
- full: prompt + completion 전체 (학습 때 한 행의 길이, 최대 길이 결정 기준)
- A 의 user 입력은 context / questionnaire / visible_state 로 나눠 각각 얼마나 차지하는지도 잰다

Qwen3 계열(0.6B ~ 8B)은 같은 토크나이저를 쓰므로 0.6B 기준 값이 8B 에도 그대로 적용된다.

실행: .venv/bin/python scripts/measure_tokens.py
결과: results/02_token_lengths.json
"""
import json
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "labeled_questionnaire_100_handoff" / "data"
OUT = ROOT / "results"
MODEL = "Qwen/Qwen3-0.6B"


def load(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values):
    values = sorted(values)
    return {"min": values[0], "median": int(median(values)), "p90": values[int(len(values) * 0.9) - 1], "max": values[-1]}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)

    def n_chat(messages, add_generation_prompt):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def n_text(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    result = {"tokenizer": MODEL, "enable_thinking": False}
    for name, fname in [("A", "extraction.jsonl"), ("B", "response.jsonl")]:
        rows = load(fname)
        prompt_len, comp_len, full_len = [], [], []
        for r in rows:
            p = n_chat(r["prompt"], add_generation_prompt=True)
            f = n_chat(r["prompt"] + r["completion"], add_generation_prompt=False)
            prompt_len.append(p)
            full_len.append(f)
            comp_len.append(n_text(r["completion"][0]["content"]))
        result[name] = {
            "prompt": stats(prompt_len),
            "completion": stats(comp_len),
            "full": stats(full_len),
            "system_prompt_tokens": n_text(rows[0]["prompt"][0]["content"]),
            "longest_id": rows[full_len.index(max(full_len))]["id"],
        }

        # user 입력을 부분별로 나눠서 측정 (JSON 으로 다시 직렬화해 각 부분 길이를 본다)
        parts = {}
        for r in rows:
            payload = json.loads(r["prompt"][1]["content"])
            for key, value in payload.items():
                parts.setdefault(key, []).append(n_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"))))
        result[name]["user_parts"] = {k: stats(v) for k, v in parts.items()}

    # 손으로 했던 대략 추정(한글 1자 = 1토큰, 그 외 3.2자 = 1토큰)과 비교
    def rough(text):
        hangul = sum(1 for ch in text if "가" <= ch <= "힣")
        return hangul + (len(text) - hangul) / 3.2

    rows = load("extraction.jsonl")
    rough_est = [rough("".join(m["content"] for m in r["prompt"])) for r in rows]
    result["A"]["rough_estimate_prompt"] = stats([int(x) for x in rough_est])

    OUT.mkdir(exist_ok=True)
    (OUT / "02_token_lengths.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
