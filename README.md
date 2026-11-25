# VoiceFocus

VoiceFocus는 혼합 WAV와 별도의 등록 WAV에서 원하는 사람의 목소리를 찾는 파일 기반 웹 데모입니다. 대화, 통화 녹음, 두 사람이 부른 무반주 화음을 같은 화면에서 처리합니다. 브라우저에서 원본과 결과를 재생하고, 파형과 스펙트로그램을 비교하고, WAV를 내려받을 수 있습니다.

## 실행

Linux CPU 환경에서 Python 3.11–3.12, Node.js 22.12 이상, Git, curl을 사용합니다. 설치 스크립트는 파일 추출, 화음 분리, 웹 통화에 쓰는 모델을 준비하고 프런트엔드를 빌드합니다.

```bash
bash scripts/setup_demo.sh
bash scripts/serve_demo.sh
```

브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000)을 열고 혼합 WAV(최대 30초)와 등록 WAV(1–20초)를 올린 후 **음성 추출**을 누르세요. 등록 음성은 **등록 음성 녹음**으로 직접 만들 수도 있습니다. 대화나 통화에는 당사자만 말한 녹음, 화음에는 당사자만 부른 녹음을 등록하세요. 파일당 최대 16 MiB를 받습니다. 자동 판별을 추가하기 전 설치에서는 `bash scripts/setup_routing.sh`로 판별 모델을 추가할 수 있습니다.

대화와 통화 녹음에서는 등록한 사람을 확인하기 어려운 구간을 음소거합니다. 같은 재생 위치에서 **원본 혼합 음성**과 **추정 음성**을 전환해 들어 보세요. [통화 평가](docs/EVALUATION.md#통화-녹음에서-등록-화자-추출)

**추출 결과 삭제**를 누르면 브라우저 비교 화면이 비워지고 서버의 결과와 비교용 원본이 즉시 삭제됩니다. [배포와 개인정보](docs/DEPLOYMENT.md)

### 실시간 마이크 실험

기존 설치에 실시간 모델을 추가하려면 아래 명령을 실행하세요.

```bash
bash scripts/setup_model.sh spk_emb_causal_100
```

localhost 또는 HTTPS에서 등록 WAV를 올리고 **실시간 마이크 실험 → 마이크 시작**을 누르세요. 입력과 처리 음량, 최근 128 ms 청크의 서버 처리 시간, 평균 RTF, 예상 재생 지연을 확인할 수 있습니다. 헤드폰을 연결하고 **처리 음성 듣기**를 켜면 분리된 마이크 음성을 들을 수 있습니다. 마이크 정지 또는 화면 이동으로 전송이 끝납니다. [지연 및 품질 측정](docs/EVALUATION.md#브라우저-마이크-실험)

### 웹 통화 실험

한 브라우저에서 **웹 통화 → 방 만들기**를 누르고, 다른 브라우저에서 같은 서버를 열어 **방 코드 → 방 참가**로 연결하세요. 같은 기기의 로컬 시험은 localhost에서, 같은 네트워크의 두 기기 시험은 HTTPS에서 진행하세요. 방에는 두 사람까지 참가할 수 있습니다.

위의 실시간 모델을 설치했다면 **상대 등록 음성**에 상대만 말한 WAV를 넣고 **VoiceFocus 켜기**를 누르세요. 처리가 시작되면 상대의 원본 재생이 음소거되고, 서버에서 검증된 음성이 수 초 늦게 재생됩니다. 두 사람 중 한 명씩 처리 음성을 시험할 수 있습니다. **VoiceFocus 끄기**로 원본 재생을 다시 켤 수 있습니다. 화면에서 나가거나 통화를 종료하면 양쪽 연결과 마이크가 해제됩니다. [시그널링과 오디오 경로](docs/ARCHITECTURE.md#웹-통화-실험)

### 화음 분리

두 사람이 함께 부르는 1–30초의 무반주 화음을 분리할 수 있습니다. 혼합 녹음과 등록 녹음 모두에 노래가 감지되면 자동으로 화음 분리를 사용합니다. **목표 목소리만**과 **목표 제거 후**를 같은 위치에서 재생하거나 WAV로 내려받을 수 있습니다. **제거 대상 바꾸기**로 두 결과의 역할을 전환할 수 있습니다.

두 사람이 같은 음을 부르거나 한쪽 목소리가 더 작으면 **남길 가수도 등록**을 열어 그 사람만 부르는 1–20초 WAV도 올려 보세요. 화음이 감지되면 두 등록 녹음을 각 분리 결과와 대조합니다. [정확도 측정](docs/EVALUATION.md#비슷한-음높이와-음량-차이)

기존 설치에 화음 모델을 추가하려면 `bash scripts/setup_harmony.sh`를 실행하세요.

일부 Linux 환경에서는 Asteroid 의존성 설치에 C 컴파일러가 사용됩니다. 이 환경에서는 `CC=gcc bash scripts/setup_harmony.sh`로 설치할 수 있습니다. CPU에서 검증했으며 모든 비교·출력 파일은 16 kHz 모노입니다. 스크립트는 UNMIXX 소스와 체크포인트를 고정 버전에서 받아오고, MIT로 공개된 Sony CSL 노래 음색 인코더와 함께 사용합니다. UNMIXX 저장소에는 별도 라이선스가 표시되어 있지 않습니다. [UNMIXX](https://github.com/jihoojung0106/unmixx/tree/8e750521b5942f4717656cac86a23cf0bd90dea5) · [노래 음색 인코더](https://huggingface.co/BernardoTorres/singer-identity)

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
  --case artifacts/librispeech/e02-full/mixed.wav artifacts/librispeech/e02-full/enroll.wav artifacts/librispeech/e02-full/target.wav

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

노래 화음 브라우저 테스트는 [합창 데이터의 개별 WAV](https://zenodo.org/records/2649950)와 위에서 설치한 화음 모델을 사용합니다.

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.05_singing.evaluate /path/to/ChoralSingingDataset artifacts/harmony
OPENBLAS_NUM_THREADS=1 python -m experiments.05_singing.evaluate_accuracy /path/to/ChoralSingingDataset artifacts/harmony-accuracy
VOICEFOCUS_E2E_HARMONY_DIR=artifacts/harmony/nino-dios-1 \
VOICEFOCUS_E2E_ACCURACY_DIR=artifacts/harmony-accuracy/rossinyol-unison python -m pytest -q frontend/tests
```

이 데모는 단일 프로세스로 로컬에서 실행하도록 구성했습니다. 동의를 확보한 음성만 올려 주세요. 업로드 원본의 서버 임시 파일은 처리 직후 삭제하며, 비교용 음성과 추출 결과는 서버 메모리에서 15분 뒤 만료됩니다. 실험 파일과 체크포인트는 Git에서 제외됩니다.
