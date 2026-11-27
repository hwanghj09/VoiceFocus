# 실험 04 | 통화 모사 및 화자 검증

## 통화 녹음에서 등록 화자 추출

ICASSP 2023 Personalized DNS Challenge는 등록 음성으로 주 화자와 주변 화자를 구별하는 조건을 평가한다. Gated Cross-Attention 연구는 목표가 말할 때의 분리 품질과 목표가 말하지 않을 때의 출력 에너지를 각각 평가한다. VoiceFocus의 통화 모드는 이 두 조건을 나눠 측정하며, REAL-TSE와 WeSpeaker로 분리 후보의 신원을 확인한 뒤 검증에 실패한 블록을 음소거한다. [ICASSP 2023 Personalized DNS Challenge](https://www.microsoft.com/en-us/research/publication/icassp-2023-deep-noise-suppression-challenge/) · [Gated Cross-Attention](https://doi.org/10.3390/electronics13112046)

LibriSpeech test-clean의 다섯 화자를 세 쌍으로 조합했다. 등록 클립과 섞은 발화는 서로 다른 녹음이다. 측정 조건은 서로 겹친 발화의 RMS 차이 6 dB, 저역 잡음이 섞인 발화, 전후에 방해 화자만 말하는 구간, 목표 화자가 전혀 없는 구간이다. 대부분의 경우 300–3400 Hz 대역과 8 kHz G.711 μ-law 변환을 거쳐 같은 16 kHz 길이의 WAV로 평가했다. 저역 잡음 혼합은 16 kHz를 사용했다. 무음은 출력 샘플이 모두 0인지 검사했고, 무음 출력의 SI-SDR은 집계에서 제외했다. [원본 녹음 · OpenSLR SLR12 · CC BY 4.0](https://www.openslr.org/12/) · [재현 방법](../../README.md#실험-재현)

| 측정 조건 | REAL-TSE 기본 | 통화 모드 |
| --- | ---: | ---: |
| G.711 겹침 `a-b-overlap`의 B 화자 SI-SDR | −19.87 dB | +10.79 dB |
| 같은 구간에서 A 화자에 대한 SI-SDR | +8.54 dB | −28.05 dB |
| `1320-3575-pauses`의 목표 무발화 구간 출력 에너지, 혼합 대비 | −2.98 dB | −22.60 dB |
| 목표 화자가 전혀 없는 파일의 완전 무음 | 0/3 | 3/3 |

목표 화자가 말한 15건 중 저역 잡음이 섞인 1건에서 목표 발화까지 모두 음소거됐다. 다른 저역 잡음 샘플에서는 무발화 구간이 무음이 됐으며 목표 SI-SDR은 +11.93 dB에서 +8.27 dB로 낮아졌다. 발화가 겹치거나 빠르게 교대하는 구간에서는 타인의 소리나 잡음이 남을 수 있다. [전체 측정 CSV](../results/calls.csv)

이 검증은 녹음 파일 전체와 1.5초 문맥을 사용한다. 실시간 자체 웹 통화에서는 같은 두 품질 지표를 지연과 함께 다시 측정해야 한다. SpeakerBeam-SS는 인과형 Conv-TasNet에 상태 공간 모델을 결합해 실시간 목표 화자 추출의 계산량을 줄였다. [SpeakerBeam-SS](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.html)

[실험 코드](../../experiments/06_calls/evaluate.py)
