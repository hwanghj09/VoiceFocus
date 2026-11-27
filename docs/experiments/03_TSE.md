# 실험 03 | 목표 화자 음성 추출

## 등록 음성과 분리 결과

| 측정 | 결과 |
| --- | --- |
| LibriSpeech 동일 화자 임베딩 코사인 유사도 중앙값, 10쌍 | 0.6691 |
| LibriSpeech 다른 화자 임베딩 코사인 유사도 중앙값, 10쌍 | 0.1167 |
| LibriSpeech 두 화자 기본 5개 혼합 × A/B 선택 SI-SDRi 중앙값 | +16.65 dB |
| 같은 문장을 읽은 합성 음성 5개 혼합 × A/B 선택 SI-SDRi 중앙값 | +10.23 dB |
| LibriSpeech 완전 겹침 E02, A 등록 및 B 등록 | +16.48 dB, +16.10 dB |
| 합성 음성 완전 겹침 E02, A 등록 및 B 등록 | +8.56 dB, +10.55 dB |

LibriSpeech E02에서 A 등록 결과는 A에 대해 16.55 dB, B에 대해 −30.09 dB의 SI-SDR을 보였다. B 등록 결과는 B에 대해 16.17 dB, A에 대해 −29.53 dB였다. 두 경우의 추출 WAV를 같은 혼합 WAV와 함께 재생할 수 있다. Chromium에서 동일한 5.025초 혼합을 업로드한 검증에서는 모델 로딩을 포함해 15.81초, 모델 추론에 4.22초가 걸렸다.

실패는 자연 음성 기본 조건에서 두 건이었다. E02의 목표가 작은 경우와 E03의 B 선택에서는 지정하지 않은 화자가 출력됐다. 합성의 부분 겹침에서 A를 선택한 두 건도 실패했다. E05의 목표가 없는 자연 음성에서는 혼합 대비 출력 에너지가 **+1.94 dB**로 간섭 화자 발화가 유지됐다. E06의 가우시안 잡음에서는 +11.73 dB, E08의 300–3400 Hz 대역 제한에서는 +16.49 dB였다. 유사 음높이, 다른 녹음 장치, 실제 전화 코덱, 3인 혼합, 독립 청취자 비교는 추가 평가가 필요하다. 잘못된 WAV, 무음, 길이 초과를 API 테스트에서 확인했다.

- 전체 파일 결과 [LibriSpeech CSV](../results/librispeech.csv) · [합성 음성 CSV](../results/synthetic.csv) · [임베딩 유사도 CSV](../results/embeddings.csv)
- 재생 파일은 평가 스크립트가 생성하는 `artifacts/eval-librispeech/e02-full/`의 `01-mixed.wav`, `02-extracted-a.wav`, `02-extracted-b.wav`를 사용한다.

## 재현

[실험 코드](../../experiments/03_tse_baseline/evaluate.py) · [실행 방법](../../README.md#실험-재현)
