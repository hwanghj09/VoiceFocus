import { useEffect, useRef, useState } from 'react'
import {
  Activity, ArrowDownToLine, AudioLines, AudioWaveform, CircleAlert, FlaskConical,
  Layers, LoaderCircle, Radio, ScanFace, ShieldCheck, SlidersHorizontal, Waves,
} from 'lucide-react'
import { extract, getAudio, getExample, getJob, getModels, mix, sleep } from './api'
import type { Job, Mode, ModelStatus } from './api'
import { AudioPreview } from './components/AudioPreview'
import { Comparison } from './components/Comparison'
import { EnrollmentRecorder } from './components/EnrollmentRecorder'
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
  const [page, setPage] = useState<'extract' | 'mix'>('extract')
  const [mode, setMode] = useState<Mode>('speech')
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
  const extraction = useRef<AbortController | null>(null)
  const singing = mode === 'harmony'
  const call = mode === 'call'
  const model = info?.models.find((entry) => entry.id === (singing ? 'harmony-vocal-split' : 'real-tse-bsrnn-spk-emb'))
  const busy = runState !== 'idle'
  const inputsLocked = busy || recordingBusy

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
    for (const audio of document.querySelectorAll('audio')) audio.pause()
    setPage(next)
  }

  function changeFile(next: File | null, type: 'mixture' | 'enroll' | 'keep') {
    if (type === 'mixture') setMixture(next)
    else if (type === 'enroll') setEnroll(next)
    else setKeepEnrollment(next)
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
      setExampleCue(cue)
      setResult(null)
    } catch (error) {
      if (!controller.signal.aborted) setExtractError(errorText(error))
    } finally {
      if (!controller.signal.aborted) setRunState('idle')
    }
  }

  async function submit() {
    if (!mixture || !enroll || !model?.available || inputsLocked) return
    const controller = new AbortController()
    extraction.current = controller
    setRunState('upload')
    setResult(null)
    setExtractError('')
    const deadline = Date.now() + (singing ? 16 : 8) * 60_000
    const timer = window.setTimeout(() => controller.abort(), deadline - Date.now())
    try {
      let job = await extract(mixture, enroll, controller.signal, mode, keepEnrollment)
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
    idle: result ? (singing ? '화음 분리 완료' : '추출 완료') : '',
    example: '예제 음성을 불러오는 중',
    upload: 'WAV 파일을 올리는 중',
    queued: '처리 대기 중',
    processing: singing ? '노래 화음을 분리하는 중' : '목표 화자를 추출하는 중',
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
          </nav>
          <div className="header-status">
            <span className={`status-dot ${model?.available ? 'online' : ''}`} />
            {model?.available ? '모델 사용 가능' : (model || modelError ? '모델 연결 필요' : '모델 확인 중')}
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="hero" aria-labelledby="intro-title">
          <div className="hero-content">
            <div className="hero-kicker"><span /> 파일 기반 데모</div>
            <h1 id="intro-title">{page === 'extract' ? (singing ? '노래 화음 분리' : call ? '통화 상대 음성 추출' : '목표 화자 추출') : '두 WAV 혼합'}</h1>
            <p>{page === 'extract'
              ? (singing ? '두 사람이 부른 화음에서 등록한 가수의 목소리를 분리합니다.' : call ? '통화 녹음에서 등록한 화자를 찾아, 확인되지 않는 구간을 음소거합니다.' : '혼합 음성에서 등록한 사람의 목소리를 추정해 분리합니다.')
              : '두 음성을 같은 시간축에 합쳐 화자 추출에 사용할 WAV를 만듭니다.'}</p>
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
              {!singing && info?.example_available && (
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
            <label className="mode-control">
              <span>분리 대상</span>
              <select value={mode} disabled={inputsLocked} onChange={(event) => {
                const nextMode = event.currentTarget.value as Mode
                setMode(nextMode)
                if (nextMode !== 'harmony') setKeepEnrollment(null)
                setResult(null)
                setExtractError('')
              }}>
                <option value="speech">일반 음성</option>
                <option value="call">통화 녹음</option>
                <option value="harmony">노래 화음</option>
              </select>
            </label>
            <div className="upload-grid">
              <Upload label="혼합 음성" hint={singing ? '두 사람이 부르는 무반주 화음 WAV · 1–30초' : call ? '통화 상황을 녹음한 WAV · 1–30초' : '여러 사람이 포함된 WAV · 최대 30초'} icon={Layers} file={mixture} disabled={inputsLocked} onChange={(next) => changeFile(next, 'mixture')} />
              <Upload label="등록 음성" hint={singing ? '빼고 싶은 사람만 부르는 별도 WAV · 1–20초' : '목표 화자 한 사람의 별도 녹음 · 1–20초'} icon={ScanFace} file={enroll} disabled={inputsLocked} onChange={(next) => changeFile(next, 'enroll')}>
                <EnrollmentRecorder active={page === 'extract'} disabled={busy} onBusy={setRecordingBusy} onRecorded={(file) => changeFile(file, 'enroll')} />
              </Upload>
            </div>
            {singing && model?.keep_enrollment_available && (
              <details className="keep-enrollment">
                <summary>남길 가수도 등록 <span>{keepEnrollment ? '등록됨' : '선택'}</span></summary>
                <Upload label="남길 가수 음성" hint="남기고 싶은 사람만 부르는 별도 WAV · 1–20초" icon={ScanFace} file={keepEnrollment} disabled={inputsLocked} onChange={(next) => changeFile(next, 'keep')} />
              </details>
            )}
            <div className="setup-actions">
              <div className="model-detail">
                <Activity size={16} aria-hidden="true" />
                <span>{model?.name || (singing ? '노래 화음 모델' : 'REAL-TSE BSRNN')} <span className="detail-separator">·</span> <span className="sample-format">16 kHz 모노</span></span>
                {(!model?.available && (info || modelError)) && <button type="button" className="button-link" onClick={() => setModelRefresh((n) => n + 1)}>다시 확인</button>}
              </div>
              <button type="button" className="primary" disabled={!mixture || !enroll || !model?.available || inputsLocked} onClick={() => void submit()}>
                {busy ? <LoaderCircle size={17} className="spin" aria-hidden="true" /> : <AudioLines size={18} aria-hidden="true" />} {singing ? '화음 분리' : '음성 추출'}
              </button>
            </div>
            {status && <p className="job-status" role="status">{status}</p>}
            {(extractError || modelError || (info && !model?.available)) && (
              <p className="error-notice" role="alert">
                <CircleAlert size={17} aria-hidden="true" />
                {extractError || modelError || (singing ? '노래 화음 모델을 설치한 뒤 다시 확인해 주세요.' : '추출하려면 서버에 REAL-TSE 체크포인트를 설치해 주세요.')}
              </p>
            )}
          </section>

          <section className="output-section" aria-labelledby="result-title">
            <div className="section-heading results-heading">
              <div className="heading-with-icon">
                <span className="section-icon"><Waves size={20} aria-hidden="true" /></span>
                <div><h2 id="result-title">원본과 결과 비교</h2><p>{singing ? '원본, 목표 목소리만 남긴 결과, 목표 목소리를 뺀 결과를 비교하세요.' : '재생 위치를 유지하면서 원본과 추정 음성을 전환할 수 있습니다.'}</p></div>
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
