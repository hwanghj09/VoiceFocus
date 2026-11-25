import { useEffect, useRef, useState } from 'react'
import {
  Activity, ArrowDownToLine, AudioLines, AudioWaveform, CircleAlert, FlaskConical,
  Layers, LoaderCircle, Phone, Radio, ScanFace, ShieldCheck, SlidersHorizontal, Waves,
} from 'lucide-react'
import { extract, getAudio, getExample, getJob, getModels, mix, sleep } from './api'
import type { Job, ModelStatus } from './api'
import { AudioPreview } from './components/AudioPreview'
import { Comparison } from './components/Comparison'
import { EnrollmentRecorder } from './components/EnrollmentRecorder'
import { LivePreview } from './components/LivePreview'
import { CallRoom } from './components/CallRoom'
import { Upload } from './components/Upload'

const BARS = [16, 21, 37, 24, 57, 86, 54, 106, 137, 76, 45, 63, 103, 67, 34, 58, 90, 145, 99, 66, 45, 76, 114, 142, 72, 44, 64, 110, 82, 40, 25, 42, 96, 67, 31, 49, 25, 14]

function errorText(error: unknown): string {
  if (error instanceof TypeError) return '서버에 연결할 수 없습니다. 연결 상태를 확인해 주세요.'
  return error instanceof Error ? error.message : '처리 중 오류가 발생했습니다.'
}

interface ExtractionResult {
  job: Job
  original: Blob
  output: Blob
  residual: Blob | null
}

export function App() {
  const [page, setPage] = useState<'extract' | 'mix' | 'call'>('extract')
  const [info, setInfo] = useState<ModelStatus | null>(null)
  const [modelError, setModelError] = useState('')
  const [modelRefresh, setModelRefresh] = useState(0)
  const [mixture, setMixture] = useState<File | null>(null)
  const [enroll, setEnroll] = useState<File | null>(null)
  const [keepEnrollment, setKeepEnrollment] = useState<File | null>(null)
  const [runState, setRunState] = useState<'idle' | 'example' | 'upload' | 'queued' | 'processing'>('idle')
  const [result, setResult] = useState<ExtractionResult | null>(null)
  const [extractError, setExtractError] = useState('')
  const [exampleCue, setExampleCue] = useState<'A' | 'B'>('A')
  const [recordingBusy, setRecordingBusy] = useState(false)
  const [liveBusy, setLiveBusy] = useState(false)
  const extraction = useRef<AbortController | null>(null)
  const ready = info?.automatic_available
  const canRegisterSecondSinger = info?.models.some((model) => model.keep_enrollment_available)
  const busy = runState !== 'idle'
  const inputsLocked = busy || recordingBusy || liveBusy

  useEffect(() => {
    const controller = new AbortController()
    getModels(controller.signal).then((status) => {
      setInfo(status)
      setModelError('')
    }).catch((error) => {
      if (!controller.signal.aborted) setModelError(errorText(error))
    })
    return () => controller.abort()
  }, [modelRefresh])

  useEffect(() => () => extraction.current?.abort(), [])

  function navigate(next: typeof page) {
    if (next === page) return
    for (const audio of document.querySelectorAll('audio')) audio.pause()
    setPage(next)
  }

  function changeFile(next: File | null, type: 'mixture' | 'enroll' | 'keep') {
    if (type === 'mixture') {
      setMixture(next)
      setKeepEnrollment(null)
    } else if (type === 'enroll') {
      setEnroll(next)
    } else {
      setKeepEnrollment(next)
    }
    setExtractError('')
    setResult(null)
  }

  async function loadExample(cue: 'A' | 'B') {
    const controller = new AbortController()
    extraction.current = controller
    setRunState('example')
    setExtractError('')
    try {
      const [input, sample] = await Promise.all([
        getExample('mixed.wav', controller.signal),
        getExample(cue === 'A' ? 'enroll.wav' : 'enroll_b.wav', controller.signal),
      ])
      if (controller.signal.aborted) return
      setMixture(input)
      setEnroll(sample)
      setKeepEnrollment(null)
      setExampleCue(cue)
      setResult(null)
    } catch (error) {
      if (!controller.signal.aborted) setExtractError(errorText(error))
    } finally {
      if (!controller.signal.aborted) setRunState('idle')
    }
  }

  async function submit() {
    if (!mixture || !enroll || !ready || inputsLocked) return
    const controller = new AbortController()
    extraction.current = controller
    setRunState('upload')
    setResult(null)
    setExtractError('')
    const deadline = Date.now() + 16 * 60_000
    const timer = window.setTimeout(() => controller.abort(), deadline - Date.now())
    try {
      let job = await extract(mixture, enroll, controller.signal, keepEnrollment)
      while (job.status === 'queued' || job.status === 'processing') {
        setRunState(job.status)
        await sleep(700, controller.signal)
        job = await getJob(job.id, controller.signal)
      }
      if (job.status === 'failed') throw new Error(job.error || '목표 화자를 추출하지 못했습니다.')
      const [original, output, residual] = await Promise.all([
        getAudio(job.id, 'original', controller.signal),
        getAudio(job.id, 'result', controller.signal),
        job.mode === 'harmony' ? getAudio(job.id, 'residual', controller.signal) : null,
      ])
      if (!controller.signal.aborted) setResult({ job, original, output, residual })
    } catch (error) {
      if (!controller.signal.aborted) setExtractError(errorText(error))
      else if (Date.now() >= deadline) setExtractError('응답 대기 시간이 초과되었습니다.')
    } finally {
      window.clearTimeout(timer)
      if (!controller.signal.aborted || Date.now() >= deadline) setRunState('idle')
    }
  }

  const status = {
    idle: result ? '추출 완료' : '',
    example: '예제 음성을 불러오는 중',
    upload: 'WAV 파일을 올리는 중',
    queued: '처리 대기 중',
    processing: '등록한 목소리를 찾는 중',
  }[runState]

  return (
    <>
      <header className="topbar">
        <div className="layout topbar-inner">
          <a href="#" className="brand" onClick={(event) => { event.preventDefault(); navigate('extract'); window.scrollTo(0, 0) }}>
            <span className="brand-icon"><AudioWaveform size={19} aria-hidden="true" /></span>
            <span>Voice<span className="brand-accent">Focus</span></span>
          </a>
          <nav className="nav" aria-label="오디오 도구">
            <button type="button" aria-current={page === 'extract' ? 'page' : undefined} onClick={() => navigate('extract')}><Radio size={16} aria-hidden="true" /> 화자 추출</button>
            <button type="button" aria-current={page === 'mix' ? 'page' : undefined} onClick={() => navigate('mix')}><Layers size={16} aria-hidden="true" /> WAV 혼합</button>
            <button type="button" aria-current={page === 'call' ? 'page' : undefined} onClick={() => navigate('call')}><Phone size={16} aria-hidden="true" /> 웹 통화</button>
          </nav>
          <div className="header-status" hidden={page === 'call'}>
            <span className={`status-dot ${ready ? 'online' : ''}`} />
            {ready ? '모델 사용 가능' : (info || modelError ? '모델 연결 필요' : '모델 확인 중')}
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="hero" aria-labelledby="intro-title">
          <div className="hero-content">
            <div className="hero-kicker"><span /> {page === 'call' ? 'WebRTC 실험' : '파일 기반 데모'}</div>
            <h1 id="intro-title">{page === 'extract' ? '목표 화자 추출' : page === 'mix' ? '두 WAV 혼합' : '웹 통화'}</h1>
            <p>{page === 'extract'
              ? '대화, 통화 녹음, 두 사람이 부른 무반주 화음에서 등록한 사람의 목소리를 찾습니다.'
              : page === 'mix' ? '두 음성을 같은 시간축에 합쳐 화자 추출에 사용할 WAV를 만듭니다.'
                : '두 브라우저를 연결하고 상대의 원본 음성과 처리 음성을 전환해 들어 보세요.'}</p>
          </div>
          <div className="hero-signal" aria-hidden="true">{BARS.map((height, index) => (
            <i key={index} style={{ height, opacity: index > 3 && index < 32 ? 0.94 : 0.48 }} />
          ))}</div>
        </section>

        <div hidden={page !== 'extract'}>
          <section className="panel setup" aria-labelledby="setup-title">
            <div className="section-heading">
              <div className="heading-with-icon">
                <span className="section-icon"><SlidersHorizontal size={20} aria-hidden="true" /></span>
                <div><h2 id="setup-title">입력 설정</h2><p>혼합 음성과 목표 화자의 등록 음성을 올려 주세요.</p></div>
              </div>
              {info?.example_available && (
                <div className="example-control">
                  <select aria-label="예제 목표 화자" disabled={inputsLocked} value={exampleCue} onChange={(event) => setExampleCue(event.currentTarget.value === 'A' ? 'A' : 'B')}>
                    <option value="A">화자 A</option>
                    <option value="B">화자 B</option>
                  </select>
                  <button type="button" className="secondary" disabled={inputsLocked} onClick={() => void loadExample(exampleCue)}>
                    <FlaskConical size={16} aria-hidden="true" /> 예제 불러오기
                  </button>
                </div>
              )}
            </div>
            <div className="upload-grid">
              <Upload label="혼합 음성" hint="대화, 통화 또는 화음 WAV · 최대 30초" icon={Layers} file={mixture} disabled={inputsLocked} onChange={(next) => changeFile(next, 'mixture')} />
              <Upload label="등록 음성" hint="분리할 사람만 말하거나 부르는 별도 WAV · 1–20초" icon={ScanFace} file={enroll} disabled={inputsLocked} onChange={(next) => changeFile(next, 'enroll')}>
                <EnrollmentRecorder active={page === 'extract'} disabled={busy || liveBusy} onBusy={setRecordingBusy} onRecorded={(file) => changeFile(file, 'enroll')} />
              </Upload>
            </div>
            {canRegisterSecondSinger && (
              <details className="keep-enrollment">
                <summary>남길 가수도 등록 <span>{keepEnrollment ? '등록됨' : '선택'}</span></summary>
                <Upload label="남길 가수 음성" hint="화음 분리에 사용하는 별도 독창 WAV · 1–20초" icon={ScanFace} file={keepEnrollment} disabled={inputsLocked} onChange={(next) => changeFile(next, 'keep')} />
              </details>
            )}
            <div className="setup-actions">
              <div className="model-detail">
                <Activity size={16} aria-hidden="true" />
                <span className="sample-format">16 kHz 모노</span>
                {(!ready && (info || modelError)) && <button type="button" className="button-link" onClick={() => setModelRefresh((n) => n + 1)}>다시 확인</button>}
              </div>
              <button type="button" className="primary" disabled={!mixture || !enroll || !ready || inputsLocked} onClick={() => void submit()}>
                {busy ? <LoaderCircle size={17} className="spin" aria-hidden="true" /> : <AudioLines size={18} aria-hidden="true" />} 음성 추출
              </button>
            </div>
            {status && <p className="job-status" role="status">{status}</p>}
            {(extractError || modelError || (info && !ready)) && (
              <p className="error-notice" role="alert">
                <CircleAlert size={17} aria-hidden="true" />
                {extractError || modelError || '자동 분리에 필요한 서버 모델을 설치해 주세요.'}
              </p>
            )}
          </section>

          {page === 'extract' && <LivePreview enrollment={enroll} available={Boolean(info?.live_available)} disabled={busy || recordingBusy} onBusy={setLiveBusy} />}

          <section className="output-section" aria-labelledby="result-title">
            <div className="section-heading results-heading">
              <div className="heading-with-icon">
                <span className="section-icon"><Waves size={20} aria-hidden="true" /></span>
                <div><h2 id="result-title">원본과 결과 비교</h2><p>같은 재생 위치에서 원본과 결과를 비교할 수 있습니다.</p></div>
              </div>
              {result && (
                <div className="metrics">
                  <span>전체 처리 <b>{((result.job.processing_ms ?? 0) / 1000).toFixed(2)}초</b></span>
                  <span>모델 추론 <b>{((result.job.inference_ms ?? 0) / 1000).toFixed(2)}초</b></span>
                  <span>{result.job.model_version}</span>
                </div>
              )}
            </div>
            {result?.job.target_muted && <p className="error-notice" aria-live="polite">등록한 화자를 확인하지 못해 추정 음성을 음소거했습니다.</p>}
            {result ? (
              <Comparison key={result.job.id} original={result.original} output={result.output} residual={result.residual} uncertainMatch={result.job.uncertain_match} inputFile={mixture} duration={result.job.duration_seconds} />
            ) : (
              <div className="preview-grid">
                <AudioPreview tone="input" label="원본 혼합 음성" description="비교 재생용 · 16 kHz 모노" blob={null} filename="voicefocus-original.wav" />
                <AudioPreview tone="output" label="추정 음성" description="등록한 목표 화자의 추출 결과" blob={null} filename="voicefocus-extracted.wav" />
              </div>
            )}
          </section>
        </div>

        {page === 'call' && <CallRoom filterAvailable={Boolean(info?.live_available)} />}

        <MixPage hidden={page !== 'mix'} canTransfer={!busy} onTransfer={(file) => {
          changeFile(file, 'mixture')
          navigate('extract')
          window.scrollTo(0, 0)
        }} />

        <footer>
          <span><ShieldCheck size={16} aria-hidden="true" /> 동의받은 음성만 사용해 주세요. 서버의 추출 결과는 처리 후 15분 뒤 만료됩니다.</span>
        </footer>
      </main>
    </>
  )
}

function MixPage({ hidden, canTransfer, onTransfer }: { hidden: boolean; canTransfer: boolean; onTransfer: (file: File) => void }) {
  const [a, setA] = useState<File | null>(null)
  const [b, setB] = useState<File | null>(null)
  const [result, setResult] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])

  function change(next: File | null, setter: (file: File | null) => void) {
    setter(next)
    setResult(null)
    setError('')
  }

  async function combine() {
    if (!a || !b || busy) return
    const request = new AbortController()
    controller.current = request
    setBusy(true)
    setResult(null)
    setError('')
    try {
      setResult(await mix(a, b, request.signal))
    } catch (cause) {
      if (!request.signal.aborted) setError(errorText(cause))
    } finally {
      if (!request.signal.aborted) setBusy(false)
    }
  }

  return (
    <div hidden={hidden}>
      <section className="panel setup" aria-labelledby="mix-title">
        <div className="section-heading">
          <div className="heading-with-icon">
            <span className="section-icon"><Layers size={20} aria-hidden="true" /></span>
            <div><h2 id="mix-title">혼합할 음성 선택</h2><p>각 WAV의 시작점을 맞춰 합칩니다.</p></div>
          </div>
        </div>
        <div className="upload-grid">
          <Upload label="화자 A 음성" hint="첫 번째 WAV · 최대 30초" icon={AudioLines} file={a} disabled={busy} onChange={(next) => change(next, setA)} />
          <Upload label="화자 B 음성" hint="두 번째 WAV · 최대 30초" icon={AudioLines} file={b} disabled={busy} onChange={(next) => change(next, setB)} />
        </div>
        <div className="setup-actions">
          <div className="model-detail"><Activity size={16} aria-hidden="true" /> 16 kHz 모노 WAV</div>
          <button type="button" className="primary" disabled={!a || !b || busy} onClick={() => void combine()}>
            {busy ? <LoaderCircle size={17} className="spin" aria-hidden="true" /> : <Layers size={17} aria-hidden="true" />} WAV 혼합
          </button>
        </div>
        {error && <p className="error-notice" role="alert"><CircleAlert size={17} aria-hidden="true" />{error}</p>}
      </section>
      <section className="output-section" aria-labelledby="mix-result-title">
        <div className="section-heading results-heading">
          <div className="heading-with-icon">
            <span className="section-icon"><Waves size={20} aria-hidden="true" /></span>
            <div><h2 id="mix-result-title">혼합 결과</h2><p>재생하거나 화자 추출 입력으로 사용하세요.</p></div>
          </div>
          {result && <button type="button" className="secondary" disabled={!canTransfer} onClick={() => onTransfer(result)}>
            <ArrowDownToLine size={16} aria-hidden="true" /> 추출 입력에 사용
          </button>}
        </div>
        <AudioPreview label="혼합 WAV" description="화자 A와 B의 합성 오디오" tone="input" blob={result} filename="voicefocus-mixed.wav" />
      </section>
    </div>
  )
}
