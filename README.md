# VoiceFocus

VoiceFocus는 혼합 음성과 별도의 등록 음성을 받아 원하는 화자의 발화를 추정하는 파일 기반 웹 데모입니다. REAL-TSE의 사전 학습 BSRNN을 사용하며, 브라우저에서 원본과 추정 음성의 재생, 파형, 스펙트로그램, WAV 다운로드를 제공합니다. WAV 혼합 도구로 직접 비교용 입력도 만들 수 있습니다.

## 실행

Python 3.11–3.12, Node.js 22.12 이상, Git과 curl이 필요합니다. 아래는 Linux CPU 환경 기준입니다. 체크포인트 묶음 다운로드에 약 1 GB, 선택한 체크포인트 설치에 약 270 MB가 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchaudio==2.7.1
bash scripts/setup_model.sh

npm --prefix frontend ci
npm --prefix frontend run build
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000)을 열고 혼합 WAV(최대 30초)와 등록 WAV(1–20초)를 올린 후 **음성 추출**을 누르세요. 등록 음성은 **등록 음성 녹음**으로 직접 만들 수 있습니다. 브라우저에서 16 kHz 모노 WAV로 변환한 뒤 **음성 추출**을 누르면 서버로 전송합니다. 등록 음성에는 목표 화자 한 사람의 별도 발화를 담아 주세요. 파일당 최대 16 MiB를 받습니다.

결과 화면의 공통 재생 위치에서 원하는 구간을 찾고 **원본 혼합 음성**과 **추정 음성**을 전환해 들어 보세요. 두 파형과 스펙트로그램은 같은 재생 위치를 표시합니다.

입력 폼에 **예제 불러오기**를 표시하려면 아래 명령으로 예제를 만드세요. 합성 예제는 eSpeak NG가 필요합니다. LibriSpeech 명령은 OpenSLR의 CC BY 4.0 테스트 발췌를 사용하며, 아래 환경 변수로 해당 예제를 서버에 표시할 수 있습니다.

```bash
python -m scripts.fetch_samples synthetic artifacts/synthetic-source
python -m experiments.01_audio_mix.mix artifacts/synthetic-source artifacts/synthetic

python -m scripts.fetch_samples librispeech artifacts/librispeech-source
python -m experiments.01_audio_mix.mix artifacts/librispeech-source artifacts/librispeech --extended
VOICEFOCUS_EXAMPLE_DIR=artifacts/librispeech/e02-full python -m uvicorn backend.app.main:app
```

개발 서버를 사용할 때는 8000번 포트에서 API를 띄운 상태로 `npm --prefix frontend run dev`를 실행하세요. API 경로는 [시스템 구조](docs/ARCHITECTURE.md)에 정리되어 있습니다.

## 실험 재현

```bash
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
```

`artifacts/eval-*/e02-full/`에 혼합 음성, A/B 등록별 추정 음성, 두 참조 음성이 저장됩니다. 오프라인 품질과 CPU 지연 측정은 [평가 결과](docs/EVALUATION.md)에서 확인할 수 있습니다. 청크 사이에 모델 상태를 유지해 지연을 줄였지만 기준을 충족하는 실시간 추출과 WebRTC는 후속 단계입니다.

## 테스트

```bash
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q backend/tests
npm --prefix frontend run build
```

실제 모델과 브라우저를 함께 검증하려면 체크포인트를 설치하고 위의 합성 예제를 생성한 뒤 아래 명령을 실행하세요. Chromium이 테스트용 WAV를 가상 마이크로 재생합니다. 시스템 Chromium을 쓰려면 `VOICEFOCUS_CHROMIUM`에 실행 파일 경로를 지정하세요.

```bash
python -m pip install -r backend/requirements-browser.txt
python -m playwright install chromium
python -m pytest -q frontend/tests
```

이 데모는 단일 프로세스로 로컬에서 실행하도록 구성했습니다. 동의를 확보한 음성만 올려 주세요. 업로드 원본의 서버 임시 파일은 처리 직후 삭제하며, 비교용 음성과 추출 결과는 서버 메모리에서 15분 뒤 만료됩니다. 실험 파일과 체크포인트는 Git에서 제외됩니다.
