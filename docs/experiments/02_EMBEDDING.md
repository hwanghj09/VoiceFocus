# 실험 02 | 화자 임베딩과 코사인 유사도

## 목적 및 방법

LibriSpeech test-clean의 화자 6930과 260을 사용했다. 각 사람의 0번 발화를 등록하고, 1–5번 독립 발화와 비교했다. 동일 화자 10쌍과 다른 화자 10쌍의 코사인 유사도를 측정했다.

| 측정 | 동일 화자 | 다른 화자 |
|---|---:|---:|
| 비교 쌍 | 10 | 10 |
| 중앙값 | **0.6691** | **0.1167** |
| 최솟값 | 0.4591 | 0.0549 |
| 최댓값 | 0.7886 | 0.2753 |

동일 화자의 최솟값이 다른 화자의 최댓값보다 높았다. 이 표본에서는 말하는 내용이 달라도 화자 특징을 구별하는 데 임베딩이 유용했다. 다만 두 화자뿐이며 일반 화자 검증 정확도나 보편적 임계값을 입증하지 않는다.

- [원본 20개 측정값 CSV](../results/embeddings.csv)
- [실험 코드](../../experiments/02_speaker_embedding/similarity.py)
- [공통 측정 환경 및 실행 방법](../EVALUATION.md)

```bash
python -m experiments.02_speaker_embedding.similarity artifacts/librispeech-source artifacts/embeddings.csv
```
