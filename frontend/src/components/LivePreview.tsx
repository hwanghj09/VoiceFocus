import { useEffect, useRef, useState } from 'react'
import { Mic, Square } from 'lucide-react'
import { INITIAL_METRICS, startVerifiedMonitor } from '../audio/verifiedMonitor'

export function LivePreview({ enrollment, available, disabled, onBusy }: {
  enrollment: File | null
  available: boolean
  disabled: boolean
  onBusy: (busy: boolean) => void
}) {
  const current = useRef<ReturnType<typeof startVerifiedMonitor> | null>(null)
  const [state, setState] = useState<'idle' | 'starting' | 'live'>('idle')
  const [monitor, setMonitor] = useState(false)
  const [error, setError] = useState('')
  const [metrics, setMetrics] = useState(INITIAL_METRICS)

  useEffect(() => () => {
    current.current?.close()
    current.current = null
    onBusy(false)
  }, [onBusy])

  function stop(message = '') {
    current.current?.close()
    current.current = null
    setState('idle')
    setError(message)
    onBusy(false)
  }

  function start() {
    if (!enrollment || current.current || disabled || !available) return
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      setError('마이크를 사용하려면 HTTPS 또는 localhost에서 열어 주세요.')
      return
    }
    document.querySelectorAll('audio').forEach((element) => element.pause())
    setError('')
    setMetrics(INITIAL_METRICS)
    setState('starting')
    onBusy(true)
    try {
      current.current = startVerifiedMonitor({
        enrollment,
        input: () => navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } }),
        ownsInput: true,
        audible: monitor,
        onReady: () => setState('live'),
        onError: stop,
        onMetrics: setMetrics,
      })
    } catch {
      stop('브라우저에서 16 kHz 오디오를 시작할 수 없습니다.')
    }
  }

  return (
    <section className="panel live-preview" aria-labelledby="live-title">
      <div className="section-heading">
        <div><h2 id="live-title">실시간 마이크 실험</h2><p>위에 등록한 목소리로 마이크 입력을 확인합니다.</p></div>
        {state === 'idle' ? (
          <button type="button" className="secondary" disabled={!available || !enrollment || disabled} onClick={() => void start()}>
            <Mic size={16} aria-hidden="true" /> 마이크 시작
          </button>
        ) : (
          <button type="button" className="secondary" onClick={() => { stop() }}>
            <Square size={15} aria-hidden="true" /> 마이크 정지
          </button>
        )}
      </div>
      {!available && <p>서버에 실시간 모델을 설치하면 사용할 수 있습니다.</p>}
      <label className="live-monitor"><input type="checkbox" checked={monitor} onChange={(event) => {
        const enabled = event.currentTarget.checked
        setMonitor(enabled)
        current.current?.setAudible(enabled)
      }} /> 처리 음성 듣기 · 헤드폰 사용</label>
      {state !== 'idle' && <p role="status">{state === 'starting' ? '실시간 모델 연결 및 마이크 권한 확인 중' : '마이크 처리 중'}</p>}
      {error && <p role="alert" className="error-notice">{error}</p>}
      {state === 'live' && <div className="live-metrics">
        <label>입력 레벨 <meter max={1} value={metrics.input} /></label>
        <label>처리 레벨 <meter max={1} value={metrics.output} /></label>
        <span>청크 처리 <b>{metrics.processing} ms</b></span>
        <span>평균 RTF <b>{metrics.rtf.toFixed(2)}</b></span>
        <span>예상 재생 지연 <b>{metrics.delay ? `${metrics.delay} ms` : '측정 중'}</b></span>
      </div>}
    </section>
  )
}
