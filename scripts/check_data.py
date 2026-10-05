"""1단계: 학습 전 데이터 점검 (표준 라이브러리만 사용)

확인하는 것
- A(extraction), B(response) 행의 키와 메시지 역할 순서
- A 정답이 JSON으로 파싱되고 최상위 키 4개를 모두 갖는지
- evidence의 context[turn_index]['content'][start:end] 가 text 와 같은지
- A, B, master 의 id 순서가 같은지
- 카테고리, 행동(action) 분포

실행: python3 scripts/check_data.py
결과: results/01_data_check.json
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "labeled_questionnaire_100_handoff" / "data"
OUT = ROOT / "results"
TOP_KEYS = ["facts", "intents", "relations", "unmapped_facts"]


def load(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_evidence(label):
    """정답 JSON 안의 모든 evidence 를 (어디서 나왔는지, evidence) 로 돌려준다."""
    for f in label["facts"]:
        yield "facts", f["evidence"]
    for i in label["intents"]:
        yield "intents", i["evidence"]
    for r in label["relations"]:
        yield "relations", r["evidence"]
    for u in label["unmapped_facts"]:
        yield "unmapped_facts", u["evidence"]


def main():
    ext, res, master = load("extraction.jsonl"), load("response.jsonl"), load("master.jsonl")
    problems = []

    ids_ok = [r["id"] for r in ext] == [r["id"] for r in res] == [r["id"] for r in master]
    if not ids_ok:
        problems.append("A, B, master 의 id 순서가 다름")

    for name, rows in [("A", ext), ("B", res)]:
        for r in rows:
            roles = [m["role"] for m in r["prompt"]] + ["|"] + [m["role"] for m in r["completion"]]
            if roles != ["system", "user", "|", "assistant"]:
                problems.append(f"{name} {r['id']} 메시지 역할 순서: {roles}")

    parsed, keys_ok = 0, 0
    ev_total, ev_match, ev_where = 0, 0, Counter()
    for r in ext:
        try:
            label = json.loads(r["completion"][0]["content"])
            parsed += 1
        except json.JSONDecodeError as e:
            problems.append(f"A {r['id']} 정답 JSON 파싱 실패: {e}")
            continue
        if list(label.keys()) == TOP_KEYS:
            keys_ok += 1
        else:
            problems.append(f"A {r['id']} 최상위 키: {list(label.keys())}")
        context = json.loads(r["prompt"][1]["content"])["context"]
        for where, ev in iter_evidence(label):
            ev_total += 1
            ev_where[where] += 1
            turn = context[ev["turn_index"]]
            if turn["role"] == "user" and turn["content"][ev["start"]:ev["end"]] == ev["text"]:
                ev_match += 1
            else:
                problems.append(f"A {r['id']} evidence 불일치 ({where}): {ev}")

    user_payload = json.loads(ext[0]["prompt"][1]["content"])
    turns = sorted(len(json.loads(r["prompt"][1]["content"])["context"]) for r in ext)

    summary = {
        "rows": {"A": len(ext), "B": len(res), "master": len(master)},
        "id_order_same": ids_ok,
        "A_label_json_parsed": parsed,
        "A_label_top_keys_ok": keys_ok,
        "evidence_total": ev_total,
        "evidence_exact_match": ev_match,
        "evidence_by_section": dict(ev_where),
        "facts_per_row": dict(sorted(Counter(len(json.loads(r["completion"][0]["content"])["facts"]) for r in ext).items())),
        "context_turns": {"min": turns[0], "median": turns[len(turns) // 2], "max": turns[-1]},
        "A_user_payload_keys": list(user_payload.keys()),
        "categories": dict(Counter(r["category"] for r in ext).most_common()),
        "actions": dict(Counter(m["plan"]["action"] for m in master).most_common()),
        "human_review": dict(Counter(r["human_review"] for r in ext)),
        "lineage_group": dict(Counter(r["lineage_group"] for r in ext)),
        "problems": problems,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "01_data_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
