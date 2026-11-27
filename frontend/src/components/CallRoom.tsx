import { useEffect, useRef, useState } from 'react'
import { Phone, PhoneOff, ScanFace } from 'lucide-react'
import { startPeerCall } from '../audio/peerCall'
import type { CallStatus } from '../audio/peerCall'
import { startVerifiedMonitor } from '../audio/verifiedMonitor'
import { Upload } from './Upload'

const STATUS_LABELS: Record<CallStatus, string> = {
  idle: '', starting: '마이크 연결 중', waiting: '상대 참가 대기 중', connecting: '상대 브라우저에 연결 중', connected: '통화 연결됨',
}

export function CallRoom({ filterAvailable }: { filterAvailable: boolean }) {
  const call = useRef<ReturnType<typeof startPeerCall> | null>(null)
  const monitor = useRef<ReturnType<typeof startVerifiedMonitor> | null>(null)
  const player = useRef<HTMLAudioElement>(null)
  const [status, setStatus] = useState<CallStatus>('idle')
  const [code, setCode] = useState('')
  const [room, setRoom] = useState('')
  const [remote, setRemote] = useState<MediaStream | null>(null)
  const [enrollment, setEnrollment] = useState<File | null>(null)
  const [filter, setFilter] = useState<'off' | 'starting' | 'on' | 'failed'>('off')
  const [delay, setDelay] = useState(0)
  const [processedLevel, setProcessedLevel] = useState(0)
  const [error, setError] = useState('')
  const [needsPlay, setNeedsPlay] = useState(false)

  useEffect(() => () => {
    call.current?.close()
    monitor.current?.close()
  }, [])

  function playRemote() {
    const element = player.current
    if (element?.srcObject) {
      void element.play().catch(() => {
        if (player.current === element && element.srcObject) setNeedsPlay(true)
      })
    }
  }

  function stop(message = '') {
    call.current?.close()
    call.current = null
    monitor.current?.close()
    monitor.current = null
    if (player.current) player.current.srcObject = null
    setStatus('idle')
    setRoom('')
    setRemote(null)
    setFilter('off')
    setNeedsPlay(false)
    setError(message)
  }

  function connect(create: boolean) {
    if (call.current) return
    if (!navigator.mediaDevices?.getUserMedia || !window.RTCPeerConnection) {
      setError('통화에 사용할 마이크를 연결하려면 HTTPS 또는 localhost에서 열어 주세요.')
      return
    }
    setError('')
    setStatus('starting')
    try {
      call.current = startPeerCall(create ? { type: 'create' } : { type: 'join', code: code.trim().toUpperCase() }, {
        onRoom: setRoom,
        onStatus: setStatus,
        onRemote: (stream) => {
          setRemote(stream)
          if (player.current) player.current.srcObject = stream
          playRemote()
        },
        onEnd: stop,
      })
    } catch {
      stop('이 브라우저에서 통화를 시작할 수 없습니다.')
    }
  }

  function toggleFilter() {
    const element = player.current
    if (!element) return
    if (filter !== 'off') {
      monitor.current?.close()
      monitor.current = null
      setFilter('off')
      setError('')
      element.muted = false
      playRemote()
      return
    }
    if (!remote || !enrollment || !filterAvailable) return
    element.muted = true
    setDelay(0)
    setProcessedLevel(0)
    setFilter('starting')
    setError('')
    function failed(message: string) {
      monitor.current = null
      setFilter('failed')
      setError(message)
    }
    try {
      monitor.current = startVerifiedMonitor({
        enrollment,
        input: async () => remote,
        ownsInput: false,
        audible: true,
        onReady: () => undefined,
        onError: failed,
        onMetrics: (metrics) => {
          if (metrics.delay > 0) {
            setDelay(metrics.delay)
            setProcessedLevel(metrics.output)
            setFilter('on')
          }
        },
      })
    } catch {
      failed('이 브라우저에서 처리 음성을 시작할 수 없습니다.')
    }
  }

  return (
    <section className="panel call-room" aria-labelledby="call-room-title">
      <div className="section-heading">
        <div><h2 id="call-room-title">통화방</h2><p>상대방에게 방 코드를 보내고 같은 네트워크에서 참가하세요.</p></div>
        {status === 'idle' ? (
          <button type="button" className="primary" onClick={() => connect(true)}><Phone size={16} aria-hidden="true" /> 방 만들기</button>
        ) : (
          <button type="button" className="secondary" onClick={() => stop()}><PhoneOff size={16} aria-hidden="true" /> 통화 나가기</button>
        )}
      </div>
      {status === 'idle' && <form className="join-call" onSubmit={(event) => { event.preventDefault(); connect(false) }}>
        <label htmlFor="join-call-code">방 코드</label>
        <input id="join-call-code" value={code} onChange={(event) => setCode(event.currentTarget.value)} autoComplete="off" maxLength={12} />
        <button className="secondary" type="submit" disabled={!/^[0-9a-f]{12}$/i.test(code.trim())}>방 참가</button>
      </form>}
      {room && <p>방 코드 <output aria-label="공유할 방 코드">{room}</output></p>}
      {status !== 'idle' && <p role="status">{STATUS_LABELS[status]}</p>}
      {error && <p role="alert" className="error-notice">{error}</p>}
      <audio ref={player} autoPlay muted={filter !== 'off'} onPlay={() => setNeedsPlay(false)} />
      {needsPlay && filter === 'off' && remote && (
        <button type="button" className="secondary" onClick={playRemote}>상대 음성 재생</button>
      )}
      <div className="call-filter">
        <Upload label="상대 등록 음성" hint="상대만 말한 별도 오디오 · 1–20초" icon={ScanFace} file={enrollment} disabled={filter !== 'off'} onChange={setEnrollment} />
        <p>VoiceFocus를 켜면 상대 음성과 등록 오디오를 이 서버에서 처리합니다. 상대의 동의를 받아 주세요. 처리 음성에는 재생 지연이 생깁니다.</p>
        {!filterAvailable && <p>서버에 실시간 모델을 설치하면 VoiceFocus를 사용할 수 있습니다.</p>}
        <button type="button" className="secondary" aria-pressed={filter !== 'off'} disabled={filter === 'off' && (!remote || !enrollment || !filterAvailable || status !== 'connected')} onClick={toggleFilter}>
          {filter === 'off' ? 'VoiceFocus 켜기' : 'VoiceFocus 끄기'}
        </button>
        {filter !== 'off' && <p aria-live="polite">{filter === 'on' ? `처리 음성 · 예상 지연 ${delay} ms` : filter === 'failed' ? '음성 처리 중단 · 재생 음소거' : '처리 음성 준비 중'}</p>}
        {filter === 'on' && <label>처리 음량 <meter max={1} value={processedLevel} /></label>}
      </div>
    </section>
  )
}
