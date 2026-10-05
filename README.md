# LoRA-A 토이 학습 (Qwen3-0.6B)

오픈잇 건강문진 에이전트 산학프로젝트의 Fine-tuning 파트에서, 우석님 100건 데이터로 정보 추출 어댑터(LoRA-A)를 작게 학습해 본 코드입니다. 목적은 학습이 돌아가고 정해진 JSON이 나오는지 확인하는 것이라, 결과 숫자는 성능 점수가 아닙니다.

데이터, 학습 결과, 어댑터는 올리지 않았습니다.

같은 데이터로 한 다른 작업
- 명찬님 LoRA-A/B 학습과 vLLM 측정: https://github.com/mungchan20/openit-finetune
- 다빈님 평가 코드(FTeval): https://github.com/Sdabin1209/FTeval

## 구성

| 파일 | 하는 일 |
|---|---|
| `scripts/check_data.py` | 키, JSON 파싱, 근거 위치(`context[turn_index][start:end] == text`), 분포 점검. 표준 라이브러리만 사용 |
| `scripts/measure_tokens.py` | Qwen3 토크나이저로 채팅 템플릿을 적용한 실제 토큰 길이 측정 (생각 모드 끔) |
| `scripts/prepare_data.py` | 문항 단위로 학습 89건, 확인용 11건 분할. 채팅 템플릿을 미리 문자열로 만들고 프롬프트와 정답 경계의 토큰이 어긋나지 않는지 확인 |
| `scripts/eval_a.py` | JSON 형식, 사실·값·상태 일치, 의도·관계 일치, 근거 문장이 원문에 있는지와 위치가 맞는지를 따로 집계. 표준 라이브러리만 사용 |
| `scripts/analyze_a.py` | 예측을 행별로 뜯어보기. 근거가 턴, 문장, 위치 중 어디서 틀렸는지 나눠 셈 |
| `scripts/build_notebook.py` | Colab 노트북 생성 |
| `notebooks/lora_a_toy_qwen3_0.6b.ipynb` | Colab 학습, 생성, 평가, 결과 묶음 내려받기 |

## 실행

```bash
# 1. 데이터 패키지를 저장소 루트에 풀기 (labeled_questionnaire_100_handoff/ 가 생김)
unzip dataset.zip

# 2. 점검과 토큰 측정, 분할 (2, 3번은 transformers 필요)
python3 scripts/check_data.py
python3 scripts/measure_tokens.py
python3 scripts/prepare_data.py

# 3. 노트북과 Colab 업로드 묶음 만들기
python3 scripts/build_notebook.py
```

Colab에서 `notebooks/lora_a_toy_qwen3_0.6b.ipynb`를 열고 GPU 런타임으로 모두 실행합니다. 업로드 셀에서 `colab_upload_A.zip`을 고르면 됩니다. `build_notebook.py`는 데이터를 노트북 안에 넣은 `*_selfcontained.ipynb`도 만드는데, 이 파일에는 데이터가 들어 있으니 올리거나 공유하지 마세요(`.gitignore`에 넣어 두었습니다).

## 설정

| 항목 | 값 |
|---|---|
| 모델 | Qwen/Qwen3-0.6B, 생각 모드 끔 |
| LoRA | rank 16, alpha 16, dropout 0.05, q·k·v·o·gate·up·down |
| 학습 | 학습률 2e-4, 배치 1 × 누적 8, 3 epoch, 정답 부분만 손실 |
| 최대 길이 | 12,288 (실측 최대 10,746토큰. TRL 기본값 1024로 두면 정답이 잘림) |

## Colab T4에서 만난 문제 (노트북에 반영됨)

1. Colab에 깔린 torchao 0.10이 최신 peft와 충돌해 ImportError가 남. 설치 셀에서 torchao를 지움
2. T4는 `torch.cuda.is_bf16_supported()`가 True를 돌려주지만 bf16 하드웨어가 없음. GPU 세대(8 이상)로 판단해 T4는 fp32 로드와 fp16 혼합정밀로 학습
3. 1만 토큰에서 어텐션 메모리 부족. Qwen3는 k·v 헤드(8)가 q(16)보다 적어 transformers가 `enable_gqa=True`로 SDPA를 부르는데, 이때 메모리 절약 커널을 쓰지 못하고 전체 행렬 경로로 빠짐. k·v를 복제해 넘기도록 패치함(계산 결과 동일)
4. 비재진입 gradient checkpointing에서 저장 텐서 개수 불일치 오류. `use_reentrant=True`로 바꿈

이 밖에 transformers 5에서는 `warmup_ratio` 대신 `warmup_steps=0.1`을 쓰고, TRL 1.14는 `completion_mask` 대신 프롬프트를 -100으로 가린 `labels`를 만듭니다.

## 결과 요약 (Colab T4, 확인용 11건)

학습 37분, 최대 GPU 메모리 4.75GB, 확인용 손실은 epoch마다 0.232, 0.149, 0.136.

| 항목 | 값 |
|---|---|
| JSON 형식 (키 4개 포함) | 11/11 |
| 사실 찾음 / 값 일치 / 상태 일치 | 86.7% / 73.3% / 73.3% |
| 정답과 완전히 같음 (위치 숫자 제외) | 4/11 |
| 근거 문장이 원문에 있음 | 86.7% |
| 근거 위치 맞음 | 1/15 |

- 근거 위치는 턴 번호와 문장은 대부분 맞고 위치 숫자만 틀렸습니다. 모델이 고른 문장을 코드가 원문에서 찾으면 13개 중 8개가 맞습니다.
- 정정, 의도, 여러 필드 문항은 거의 못 배웠습니다. 이 유형은 100건 중 2~6건뿐이지만, 0.6B가 작은 모델이라 원인은 큰 모델로 다시 확인해야 합니다.
- 100건이 모두 같은 문장 틀에서 나온 데이터라 확인용 점수는 일반화 성능이 아닙니다.
- 위 숫자는 이 저장소의 `eval_a.py` 기준입니다. FTeval 기준 점수는 아직 맞추지 않았습니다.
