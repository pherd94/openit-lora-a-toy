"""LoRA-A 예측을 행별로 뜯어보기 (표준 라이브러리만 사용)

- 근거(evidence)가 어떻게 틀렸는지 나눠 센다
  turn_index 가 정답과 같은지 / text 가 정답과 같은지 / text 가 그 턴 원문에 있는지 / start·end 가 맞는지
  start 가 0 이나 정답과 같은 숫자처럼 외운 값인지도 본다
- 행마다 사례 종류, 맞은 것과 틀린 것을 한 줄로 요약한다

실행: python3 scripts/analyze_a.py results/colab_t4_2026-10-04
결과: 같은 폴더의 analysis.json, 화면 출력
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_a import parse_output  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PREP = ROOT / "prepared"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def all_evidence(obj):
    out = []
    for section in ["facts", "intents", "relations", "unmapped_facts"]:
        for e in obj.get(section) or []:
            if isinstance(e, dict) and isinstance(e.get("evidence"), dict):
                key = (e.get("item_id"), e.get("field") or e.get("intent") or e.get("relation"))
                out.append((section, key, e["evidence"]))
    return out


def analyze(pred_path, gold_rows):
    gold_by_id = {g["id"]: g for g in gold_rows}
    ev_counter, rows = Counter(), []
    for p in load_jsonl(pred_path):
        g = gold_by_id[p["id"]]
        gold = json.loads(g["gold"])
        context = json.loads(g["messages"][1]["content"])["context"]
        obj, how = parse_output(p["output"])
        row = {"id": p["id"], "category": g["category"], "item_id": g["item_id"], "parse": how,
               "new_tokens": p.get("new_tokens"), "prompt_tokens": p.get("prompt_tokens")}
        if not isinstance(obj, dict):
            row["note"] = "JSON 아님: " + p["output"][:120].replace("\n", " ")
            rows.append(row)
            continue
        gold_ev = {(s, k): e for s, k, e in all_evidence(gold)}
        for s, k, e in all_evidence(obj):
            ge = gold_ev.get((s, k))
            ev_counter["pred_evidence"] += 1
            turn = context[e["turn_index"]] if isinstance(e.get("turn_index"), int) and 0 <= e["turn_index"] < len(context) else None
            in_turn = bool(turn and turn["role"] == "user" and e.get("text") and e["text"] in turn["content"])
            ev_counter["text_in_claimed_turn"] += in_turn
            ev_counter["text_in_last_user_turn"] += bool(e.get("text") and e["text"] in context[-1]["content"])
            if ge is None:
                ev_counter["no_matching_gold"] += 1
                continue
            ev_counter["matched_gold"] += 1
            ev_counter["turn_index_ok"] += e.get("turn_index") == ge["turn_index"]
            ev_counter["text_exact_ok"] += e.get("text") == ge["text"]
            ev_counter["start_ok"] += e.get("start") == ge["start"]
            ev_counter["end_ok"] += e.get("end") == ge["end"]
            ev_counter["start_is_zero"] += e.get("start") == 0
            ev_counter["gold_start_is_zero"] += ge["start"] == 0
            if in_turn:  # 문장은 맞게 골랐는데 위치만 틀렸는지
                real_start = turn["content"].find(e["text"])
                ev_counter["text_found_offset_would_be_right"] += real_start == ge["start"] and e["turn_index"] == ge["turn_index"]
        pred_keys = {(f.get("item_id"), f.get("field")): f for f in obj.get("facts") or [] if isinstance(f, dict)}
        wrong = []
        for gf in gold["facts"]:
            pf = pred_keys.get((gf["item_id"], gf["field"]))
            if pf is None:
                wrong.append(f"빠짐 {gf['item_id']}.{gf['field']}")
            elif pf.get("value") != gf["value"]:
                wrong.append(f"값 {gf['item_id']}.{gf['field']} 정답={gf['value']} 예측={pf.get('value')}")
            elif pf.get("semantic_status") != gf["semantic_status"]:
                wrong.append(f"상태 {gf['item_id']}.{gf['field']} 정답={gf['semantic_status']} 예측={pf.get('semantic_status')}")
        extra = [f"{k[0]}.{k[1]}" for k in pred_keys if k not in {(f["item_id"], f["field"]) for f in gold["facts"]}]
        if extra:
            wrong.append("추가 " + ", ".join(extra))
        gi = {(i["item_id"], i["intent"]) for i in gold["intents"]}
        pi = {(i.get("item_id"), i.get("intent")) for i in obj.get("intents") or [] if isinstance(i, dict)}
        if gi != pi:
            wrong.append(f"의도 정답={sorted(gi)} 예측={sorted(pi)}")
        gr = {(r["item_id"], r["relation"]) for r in gold["relations"]}
        pr = {(r.get("item_id"), r.get("relation")) for r in obj.get("relations") or [] if isinstance(r, dict)}
        if gr != pr:
            wrong.append(f"관계 정답={sorted(gr)} 예측={sorted(pr)}")
        row["wrong"] = wrong
        row["user_text"] = context[-1]["content"]
        rows.append(row)
    return {"evidence": dict(ev_counter), "rows": rows}


def main():
    run = Path(sys.argv[1])
    val, train = load_jsonl(PREP / "A_val.jsonl"), load_jsonl(PREP / "A_train.jsonl")
    result = {}
    for name, gold in [("val_lora", val), ("train_lora", train), ("val_base", val)]:
        result[name] = analyze(run / f"pred_{name}.jsonl", gold)
    (run / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ["val_lora", "train_lora"]:
        print(f"== {name} 근거 분석:", result[name]["evidence"])
        for r in result[name]["rows"]:
            print(f"  {r['id']} {r['category']:22s} {r['item_id']:7s} tok={r.get('new_tokens')} | "
                  + ("; ".join(r.get("wrong") or []) or r.get("note") or "fact·의도·관계 모두 맞음"))
    print("== val_base 첫 출력:", load_jsonl(run / "pred_val_base.jsonl")[0]["output"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
