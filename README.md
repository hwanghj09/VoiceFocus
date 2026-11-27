# VoiceFocus

> 여러 사람이 동시에 말할 때, 등록한 사람의 목소리만 추출하는 AI 음성 분리 연구 및 웹 애플리케이션.

**파동의 중첩 → FFT → STFT → 화자 임베딩 → REAL-TSE → 에너지·화자 검증 → 실시간 통화**

## 연구 최종 정리

서로 다른 주파수의 사인파는 FFT로 분리할 수 있지만 실제 목소리는 주파수 대역이 시간에 따라 변하고 서로 겹친다. 목표 화자의 별도 등록 음성에서 특징을 추출하고, 사전 학습된 REAL-TSE 모델로 목표 음성을 추정한다. 혼합 신호의 에너지와 임베딩 유사도 기반 검증으로 잘못된 화자 선택 및 목표 부재 구간의 출력 문제를 일부 개선했다.

| 핵심 실험 | 결과 |
|---|---:|
| 440/880 Hz 파동 복원 RMSE | 3.008×10⁻¹⁴ / 5.976×10⁻¹⁴ |
| 동일/다른 화자 코사인 유사도 중앙값 | 0.6691 / 0.1167 |
| 자연 음성 SI-SDRi 중앙값 | +16.65 dB |
| 합성 음성 SI-SDRi 중앙값 | +10.23 dB |
| 통화 모사, 목표 부재 시 완전 음소거 | 0/3 → 3/3 |
| 적응형 검증, E02 출력 지연 p95 | 1.58 / 1.73초 |

측정값은 기존 2026-11-25 AI 실험과 2026-11-27 DSP 재현 계산을 구별한다. CPU 시뮬레이션에서 100 ms 미만 지연 목표는 미달했다. 일반 전화 앱 연동 및 실제 전화망 전체 지연은 검증하지 않았다.

### 연구 문서

| 문서 | 내용 |
|---|---|
| [연구 과정](docs/RESEARCH_PROCESS.md) | 연구 동기, 단계별 원리, 문제 해결 |
| [최종 연구 결과](docs/EVALUATION.md) | 핵심 수치, 결론 및 한계 |
| [01 파동·FFT·STFT](docs/experiments/01_DSP.md) | 기초 실험, 복원 오차 |
| [02 화자 임베딩](docs/experiments/02_EMBEDDING.md) | 코사인 유사도 비교 |
| [03 목표 화자 추출](docs/experiments/03_TSE.md) | 자연·합성 음성 분리 |
| [04 통화 검증](docs/experiments/04_CALL.md) | 잔차·에너지·유사도 검증 |
| [05 스트리밍](docs/experiments/05_STREAMING.md) | 독립 청크와 상태 유지 |
| [06 실시간 개선](docs/experiments/06_LIVE.md) | 검증 창, 지연·품질 |
| [원본 실험 데이터](docs/results/) | CSV, JSON |
| [시스템 구조](docs/ARCHITECTURE.md) · [배포](docs/DEPLOYMENT.md) | API, 설치 및 개인정보 |

## 주요 기능

- **화자 추출:** 혼합 음성(최대 30초)과 등록 음성(1–20초)을 파일로 업로드하거나 마이크로 녹음
- **WAV 혼합:** 화자 A/B 음성을 각각 업로드 또는 녹음하고 합성한 뒤 추출 화면으로 전달
- **결과 비교:** 같은 시간 위치에서 원본과 결과 재생, 파형·스펙트로그램, 다운로드
- **실시간 마이크:** 등록된 화자의 음성 추정과 지연 표시
- **웹 통화:** WebRTC 자체 통화방, 상대 등록 음성으로 수신 음성 필터링

## 실행

### Windows (PowerShell)

Python 3.11 또는 3.12 x64, Node.js 22.12 이상, Git for Windows, curl.exe가 필요하다. 신규 설치:

```powershell
git pull --ff-only
powershell -ExecutionPolicy Bypass -File scripts/setup_demo.ps1
powershell -ExecutionPolicy Bypass -File scripts/serve_demo.ps1
```

기존 설치는 서버 종료 후 모델 재다운로드 없이 업데이트한다.

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
npm.cmd --prefix frontend run build
powershell -ExecutionPolicy Bypass -File scripts/serve_demo.ps1
```

Windows 실제 모델 추론은 별도 확인이 필요하다. 의존성 설치 시 Visual Studio Build Tools가 필요할 수 있다.

### Linux

Python 3.11–3.12, Node.js 22.12 이상, Git, curl 필요.

```bash
bash scripts/setup_demo.sh
bash scripts/serve_demo.sh
```

기존 설치:

```bash
git pull --ff-only
.venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend run build
bash scripts/serve_demo.sh
```

브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000)을 연다. 마이크는 localhost 또는 HTTPS가 필요하다. 혼합·등록 음성은 WAV, MP3, M4A, AAC, MP4, WebM, FLAC, Ogg/Opus 등 실제 FFmpeg 빌드가 지원하는 형식을 사용할 수 있다. 파일당 최대 16 MiB, 16 kHz 모노 변환을 사용한다. 영상은 첫 오디오 트랙을 사용한다.

## 사용 방법

1. **화자 추출:** 혼합 음성과 목표 화자만 말한 등록 음성을 입력하고 음성 추출을 누른다. 화면에서 직접 녹음할 수도 있다.
2. **WAV 혼합:** 두 사람의 음성을 각각 녹음 또는 업로드한 후 혼합한다. 추출 입력에 사용을 누르면 혼합 파일이 전달된다.
3. **실시간 마이크:** 등록 음성을 선택한 뒤 마이크를 시작한다. 헤드폰을 착용한 뒤 처리 음성을 듣는다.
4. **웹 통화:** 두 브라우저가 같은 서버를 사용해 방을 만들고 참가한다. 상대 등록 음성을 선택해 VoiceFocus를 켠다. 동일 네트워크에서 HTTPS를 권장한다.

기본 모델 설치 상태는 `/api/v1/models`에서 확인할 수 있다. 실시간 체크포인트만 추가하려면 `bash scripts/setup_model.sh spk_emb_causal_100` (Windows는 `scripts/setup_model.ps1 -Model spk_emb_causal_100`)을 사용한다.

## 예제 및 개발

입력 폼에 **예제 불러오기**를 표시하려면 아래 명령으로 예제를 만드세요. 합성 예제는 eSpeak NG가 필요합니다. LibriSpeech 명령은 OpenSLR의 CC BY 4.0 테스트 발췌를 사용하며, 아래 환경 변수로 해당 예제를 서버에 표시할 수 있습니다. 설치 스크립트로 만든 가상 환경은 `source .venv/bin/activate`로 활성화할 수 있습니다.

```bash
python -m scripts.fetch_samples synthetic artifacts/synthetic-source
python -m experiments.01_audio_mix.mix artifacts/synthetic-source artifacts/synthetic

python -m scripts.fetch_samples librispeech artifacts/librispeech-source
python -m experiments.01_audio_mix.mix artifacts/librispeech-source artifacts/librispeech --extended
VOICEFOCUS_EXAMPLE_DIR=artifacts/librispeech/e02-full bash scripts/serve_demo.sh
```

개발 서버를 사용할 때는 8000번 포트에서 API를 띄운 상태로 `npm --prefix frontend run dev`를 실행하세요. API 경로는 [시스템 구조](docs/ARCHITECTURE.md)에 정리되어 있습니다.

## 실험 재현

```bash
python -m experiments.00_dsp.evaluate
export OPENBLAS_NUM_THREADS=1
python -m experiments.02_speaker_embedding.similarity artifacts/librispeech-source artifacts/embeddings.csv
python -m experiments.03_tse_baseline.evaluate artifacts/librispeech artifacts/eval-librispeech
python -m experiments.03_tse_baseline.evaluate artifacts/synthetic artifacts/eval-synthetic
python -m experiments.01_audio_mix.plot artifacts/comparison.png \
  artifacts/eval-librispeech/e02-full/01-mixed.wav \
  artifacts/eval-librispeech/e02-full/02-extracted-a.wav \
  artifacts/eval-librispeech/e02-full/02-extracted-b.wav

bash scripts/setup_model.sh spk_emb_causal_100
python -m experiments.04_streaming.benchmark artifacts/librispeech/e02-full artifacts/streaming.json
python -m experiments.04_streaming.benchmark_stateful \
  artifacts/librispeech/e02-full artifacts/streaming-stateful.json
python -m experiments.04_streaming.benchmark_live artifacts/streaming-live.json \
  --case artifacts/librispeech/e02-full/mixed.wav artifacts/librispeech/e02-full/enroll.wav artifacts/librispeech/e02-full/target.wav \
  --case artifacts/librispeech/e02-full/mixed.wav artifacts/librispeech/e02-full/enroll_b.wav artifacts/librispeech/e02-full/interferer.wav

python -m scripts.fetch_samples librispeech-calls artifacts/calls-source
python -m experiments.06_calls.evaluate artifacts/calls-source artifacts/calls \
  --pair a b --pair 1320 3575 --pair 8455 b
```

`artifacts/eval-*/e02-full/`에 혼합 음성, A/B 등록별 추정 음성, 두 참조 음성이 저장됩니다. 오프라인 품질과 CPU 지연 측정은 [평가 결과](docs/EVALUATION.md)에서 확인할 수 있습니다.

## 테스트

```bash
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q backend/tests
npm --prefix frontend run build
```

실제 모델과 브라우저를 함께 검증하려면 체크포인트를 설치하고 위의 합성 예제를 생성한 뒤 아래 명령을 실행하세요. Chromium이 테스트용 WAV를 가상 마이크로 재생합니다. 시스템 Chromium을 쓰려면 `VOICEFOCUS_CHROMIUM`에 실행 파일 경로를 지정하세요. 웹 통화 테스트에는 브라우저 간 WebRTC 연결이 허용된 환경이 필요합니다.

```bash
python -m pip install -r backend/requirements-browser.txt
python -m playwright install chromium
python -m pytest -q frontend/tests
```

통합 흐름에서 통화의 무음과 겹친 발화에서의 화자 선택을 검증하려면 통화 평가를 실행한 후 아래 환경 변수를 지정하세요.

```bash
VOICEFOCUS_E2E_CALL_DIR=artifacts/calls/8455-b-absent \
VOICEFOCUS_E2E_CALL_OVERLAP_DIR=artifacts/calls/a-b-overlap python -m pytest -q frontend/tests
```

## 개인정보와 한계

동의받은 음성만 사용한다. 서버 임시 파일은 정상 작업 종료 시 삭제하고 원본 비교용 및 결과 WAV는 메모리에서 15분 뒤 만료된다. 추출 결과 삭제 기능으로 즉시 삭제할 수 있다. 외부 공개 배포 전에는 인증과 HTTPS를 구성한다. [배포 안내](docs/DEPLOYMENT.md)
