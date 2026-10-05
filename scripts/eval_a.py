"""5단계: LoRA-A 출력 평가 (표준 라이브러리만 사용, Colab 과 Mac 모두에서 실행)

입력
- 예측 파일: 한 줄에 {"id": ..., "output": 모델 출력 문자열, "ttft_s": ..., "total_s": ..., "new_tokens": ...}
- 정답 파일: prepared/A_val.jsonl 등 (id, messages, gold 가 있는 파일)

집계 항목 (작업 노트 '토이 학습 확인 항목' 기준)
- JSON 형식: 파싱 성공 비율, 최상위 키 4개가 모두 있는 비율
- 추출 값: 정답 fact 의 (item_id, field) 를 찾은 비율, 그 중 value 까지 같은 비율, 정답에 없는 fact 수
- 상태 값: semantic_status, precision 일치, intents 의 (item_id, intent) 일치, relations 일치
- 근거: evidence text 가 그 턴 원문에 실제로 있는 비율, start/end 로 잘랐을 때 text 와 같은 비율 (따로 집계)
- 행 전체가 정답과 완전히 같은 비율
- 속도: 첫 토큰까지 시간, 전체 응답 시간 (예측 파일에 있을 때)

실행: python3 scripts/eval_a.py --pred 예측.jsonl --gold prepared/A_val.jsonl [--out 결과.json]
"""
import argparse
import json
import re
from statistics import median

TOP_KEYS = ["facts", "intents", "relations", "unmapped_facts"]


def parse_output(text):
    """모델 출력에서 JSON 객체를 꺼낸다. 남아 있는 <think> 블록이나 앞뒤 잡음은 걷어낸다."""
    try:
        return json.loads(text.strip()), "clean"
    except json.JSONDecodeError:
        pass
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1]), "extracted"
        except json.JSONDecodeError:
            pass
    return None, "fail"


def evidences(obj):
    for section in TOP_KEYS:
        for entry in obj.get(section) or []:
            if isinstance(entry, dict) and isinstance(entry.get("evidence"), dict):
                yield entry["evidence"]


def check_evidence(ev, context):
    """(원문에 text 가 있는가, start/end 가 맞는가)"""
    try:
        turn = context[ev["turn_index"]]
    except (KeyError, IndexError, TypeError):
        return False, False
    if turn.get("role") != "user" or not isinstance(ev.get("text"), str) or not ev["text"]:
        return False, False
    source = turn["content"]
    in_source = ev["text"] in source
    try:
        offset_ok = source[ev["start"]:ev["end"]] == ev["text"]
    except (KeyError, TypeError):
        offset_ok = False
    return in_source, offset_ok


def fact_key(f):
    return (f.get("item_id"), f.get("field"))


def ratio(num, den):
    return None if den == 0 else round(num / den, 4)


def evaluate(preds, golds):
    gold_by_id = {g["id"]: g for g in golds}
    c = {k: 0 for k in [
        "rows", "parsed", "parsed_clean", "top_keys_ok", "exact_row",
        "gold_facts", "fact_found", "value_ok", "status_ok", "precision_ok", "extra_facts",
        "gold_intents", "intent_ok", "pred_intents", "gold_relations", "relation_ok", "pred_relations",
        "pred_evidence", "ev_in_source", "ev_offset_ok",
    ]}
    per_row, ttft, total = [], [], []
    for p in preds:
        g = gold_by_id[p["id"]]
        gold = json.loads(g["gold"])
        context = json.loads(g["messages"][1]["content"])["context"]
        obj, how = parse_output(p["output"])
        c["rows"] += 1
        if p.get("ttft_s") is not None:
            ttft.append(p["ttft_s"])
        if p.get("total_s") is not None:
            total.append(p["total_s"])
        row = {"id": p["id"], "category": g.get("category"), "item_id": g.get("item_id"), "parse": how}
        c["gold_facts"] += len(gold["facts"])
        c["gold_intents"] += len(gold["intents"])
        c["gold_relations"] += len(gold["relations"])
        if not isinstance(obj, dict):
            per_row.append(row)
            continue
        c["parsed"] += 1
        c["parsed_clean"] += how == "clean"
        c["top_keys_ok"] += all(isinstance(obj.get(k), list) for k in TOP_KEYS)
        c["exact_row"] += obj == gold
        row["exact"] = obj == gold

        pred_facts = [f for f in obj.get("facts") or [] if isinstance(f, dict)]
        pred_by_key = {fact_key(f): f for f in pred_facts}
        gold_keys = {fact_key(f) for f in gold["facts"]}
        c["extra_facts"] += sum(1 for k in pred_by_key if k not in gold_keys)
        for gf in gold["facts"]:
            pf = pred_by_key.get(fact_key(gf))
            if pf is None:
                continue
            c["fact_found"] += 1
            c["value_ok"] += pf.get("value") == gf["value"]
            c["status_ok"] += pf.get("semantic_status") == gf["semantic_status"]
            c["precision_ok"] += pf.get("precision") == gf["precision"]

        pred_intents = {(i.get("item_id"), i.get("intent")) for i in obj.get("intents") or [] if isinstance(i, dict)}
        c["pred_intents"] += len(pred_intents)
        c["intent_ok"] += sum(1 for i in gold["intents"] if (i["item_id"], i["intent"]) in pred_intents)

        def rel_key(r):
            return (r.get("item_id"), r.get("relation"), json.dumps(r.get("correction_target"), sort_keys=True))
        pred_rel = {rel_key(r) for r in obj.get("relations") or [] if isinstance(r, dict)}
        c["pred_relations"] += len(pred_rel)
        c["relation_ok"] += sum(1 for r in gold["relations"] if rel_key(r) in pred_rel)

        for ev in evidences(obj):
            c["pred_evidence"] += 1
            in_source, offset_ok = check_evidence(ev, context)
            c["ev_in_source"] += in_source
            c["ev_offset_ok"] += offset_ok
        per_row.append(row)

    metrics = {
        "rows": c["rows"],
        "json_parse_rate": ratio(c["parsed"], c["rows"]),
        "json_parse_without_cleanup_rate": ratio(c["parsed_clean"], c["rows"]),
        "top_keys_ok_rate": ratio(c["top_keys_ok"], c["rows"]),
        "exact_row_match_rate": ratio(c["exact_row"], c["rows"]),
        "fact_found_rate": ratio(c["fact_found"], c["gold_facts"]),
        "fact_value_match_rate": ratio(c["value_ok"], c["gold_facts"]),
        "fact_status_match_rate": ratio(c["status_ok"], c["gold_facts"]),
        "fact_precision_match_rate": ratio(c["precision_ok"], c["gold_facts"]),
        "extra_facts": c["extra_facts"],
        "intent_match_rate": ratio(c["intent_ok"], c["gold_intents"]),
        "relation_match_rate": ratio(c["relation_ok"], c["gold_relations"]),
        "evidence_text_in_source_rate": ratio(c["ev_in_source"], c["pred_evidence"]),
        "evidence_offset_correct_rate": ratio(c["ev_offset_ok"], c["pred_evidence"]),
        "counts": c,
    }
    if ttft:
        metrics["ttft_s_median"] = round(median(ttft), 3)
    if total:
        metrics["total_s_median"] = round(median(total), 3)
        metrics["total_s_max"] = round(max(total), 3)
    return {"metrics": metrics, "per_row": per_row}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    result = evaluate(load_jsonl(args.pred), load_jsonl(args.gold))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
