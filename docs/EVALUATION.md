# VoiceFocus 최종 연구 결과

> **결론:** 주파수만으로는 실제 혼합 음성에서 목표 화자를 유일하게 결정하기 어렵다. 화자 임베딩과 AI 모델, 파형 에너지 기반 검증을 결합해 특정 화자의 음성을 추정할 수 있었지만 실시간 지연과 오류는 남아 있다.

## 체크포인트와 재현 환경

파일 데모에 REAL-TSE 공식 `spk_emb_100`을 사용했다. 주 WeSep 저장소의 최신 버전은 사전 학습 모델 배포를 준비 중이다. REAL-TSE의 소스는 `d9ed77d`, WeSpeaker는 `8f53b64`에서 고정했다. ZIP의 SHA-256을 검사한 뒤 필요한 체크포인트만 추출한다. 추론 코드 및 REAL-TSE 학습 파라미터의 Apache-2.0 조건과, 내장 ECAPA 파라미터에 적용되는 별도 업스트림 조건은 각 프로젝트 문서를 참조한다. [WeSep](https://github.com/wenet-e2e/wesep) · [REAL-TSE 및 체크포인트 조건](https://github.com/REAL-TSE/wesep-real-tse#2-download-pretrained-model) · [WeSpeaker](https://github.com/wenet-e2e/wespeaker)

2026-11-25에 Python 3.12, PyTorch CPU 2.7.1, Intel Xeon Platinum 8573C, CPU 할당량 2개, 추론 스레드 2개에서 측정했다. 자연 음성은 LibriSpeech test-clean 화자 6930과 260의 독립 발화 6개씩을 사용했다. 합성 음성은 eSpeak NG 1.52.0의 두 음성으로 같은 문장을 읽게 했다. LibriSpeech는 OpenSLR에서 CC BY 4.0으로 제공하며, 원본과 파생 음성은 이 저장소에 보관하지 않는다. 두 화자 표본이므로 일반화 성능을 나타내지 않는다. [OpenSLR SLR12](https://www.openslr.org/12/) · [실험 재현](../README.md#실험-재현)

## 핵심 결과 요약

| 연구 단계 | 핵심 결과 | 상세 기록 |
|---|---|---|
| 파동·FFT·STFT | 440/880 Hz 복원 RMSE 약 3.0×10⁻¹⁴ / 6.0×10⁻¹⁴ | [실험 01](experiments/01_DSP.md) |
| 화자 특징 | 동일/타 화자 유사도 중앙값 0.6691 / 0.1167 | [실험 02](experiments/02_EMBEDDING.md) |
| 실제 음성 추출 | 자연/합성 음성 SI-SDRi 중앙값 +16.65 / +10.23 dB | [실험 03](experiments/03_TSE.md) |
| 통화 모사 | 목표 부재 시 완전 음소거 0/3 → 3/3 | [실험 04](experiments/04_CALL.md) |
| 스트리밍 | 상태 유지 64 ms 청크 지연 p95 최저 158.07 ms (품질 검증 전) | [실험 05](experiments/05_STREAMING.md) |
| 적응형 검증 | E02 겹침 A/B 출력 지연 p95 1.58 / 1.73초 | [실험 06](experiments/06_LIVE.md) |

기초 DSP 수치는 2026-11-27에 NumPy/SciPy로 재현한 계산값이고, AI 성능 수치는 기존 2026-11-25 평가 기록이다. 모델·데이터 조건을 달리한 실험의 수치를 같은 성능으로 해석하지 않는다.

## 연구 결론

- 이상적인 사인파에서는 직교성을 이용한 주파수 분리와 복원이 가능했다.
- 실제 음성에서는 주파수가 시간에 따라 변하고 화자끼리 겹친다. 등록 음성의 특징을 조건으로 하는 REAL-TSE 모델을 활용했다.
- 통화 모사에서 잘못 선택한 화자를 잔차와 코사인 유사도로 다시 검증한 결과 일부 조건에서 정확도가 개선됐다. 특정 무발화 구간의 출력 에너지는 혼합 대비 −2.98에서 −22.60 dB로 줄었다.
- 웹 프로그램에는 음성 파일·마이크 입력, 혼합, 추출, 원본/결과 비교, 자체 WebRTC 통화를 구현했다. 독립 실시간 마이크 실험의 측정 기록은 유지하지만 최종 웹 화면에서는 해당 기능을 제공하지 않는다.

## 한계와 후속 연구

- 자연 음성 기본 조건과 합성 음성 부분 겹침에서 잘못된 화자 추출 사례가 있었다.
- 목표 화자가 존재하는 저역 잡음 조건에서도 잘못 음소거한 경우가 있었다.
- 100 ms 미만 추가 지연 및 RTF 0.5 미만이라는 목표는 달성하지 못했다. CPU 시뮬레이션 지연에는 실제 네트워크·스피커 지연이 포함되지 않는다.
- 다양한 화자·장치·잡음·3인 이상 혼합, 실제 두 기기 간 통화 품질과 사용자 청취 평가가 추가로 필요하다.

## 자료 및 재현

[연구 과정](RESEARCH_PROCESS.md) · [실험 원본 데이터](results/) · [시스템 구조](ARCHITECTURE.md) · [실행 및 재현 방법](../README.md)

- [REAL-TSE](https://github.com/REAL-TSE/wesep-real-tse)
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
- [LibriSpeech](https://www.openslr.org/12/)
- [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)
- [SciPy STFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html)
