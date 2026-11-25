import { useEffect, useRef, useState } from 'react'
import { Mic, Square } from 'lucide-react'

const RATE = 16000
const CHUNK = 2048
const MAX_PENDING = 8

interface Session {
  audio: AudioContext
  socket: WebSocket
  stream?: MediaStream
  capture?: AudioWorkletNode
  timeout?: number
  monitor: GainNode
  pending: { pcm: ArrayBuffer; captured: number }[]
  sent: { remaining: number; captured: number }[]
  awaiting: boolean
  playAt: number
  inputLevel: number
  computeMs: number
  processedChunks: number
}

function dispose(session: Session) {
  window.clearTimeout(session.timeout)
  session.socket.onopen = null
  session.socket.onclose = null
  session.socket.onmessage = null
  session.socket.close()
  session.capture?.disconnect()
  session.capture?.port.close()
  session.stream?.getTracks().forEach((track) => track.stop())
  void session.audio.close().catch(() => undefined)
}

function level(samples: Float32Array) {
  let squares = 0
  for (const sample of samples) squares += sample * sample
  return Math.min(1, Math.sqrt(squares / samples.length) * 4)
}

export function LivePreview({ enrollment, available, disabled, onBusy }: {
  enrollment: File | null
  available: boolean
  disabled: boolean
  onBusy: (busy: boolean) => void
}) {
  const current = useRef<Session | null>(null)
  const [state, setState] = useState<'idle' | 'starting' | 'live'>('idle')
  const [monitor, setMonitor] = useState(false)
  const [error, setError] = useState('')
  const [metrics, setMetrics] = useState({ input: 0, output: 0, delay: 0, processing: 0, rtf: 0 })

  useEffect(() => () => {
    const session = current.current
    current.current = null
    if (session) dispose(session)
    onBusy(false)
  }, [onBusy])

  function stop(session: Session, message = '') {
    if (current.current !== session) return
    current.current = null
    dispose(session)
    setState('idle')
    setError(message)
    onBusy(false)
  }

  async function start() {
    if (!enrollment || current.current || disabled || !available) return
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      setError('마이크를 사용하려면 HTTPS 또는 localhost에서 열어 주세요.')
      return
    }
    let audio: AudioContext
    try {
      audio = new AudioContext({ sampleRate: RATE })
    } catch {
      setError('브라우저에서 16 kHz 오디오를 시작할 수 없습니다.')
      return
    }
    const url = new URL('/ws/v1/stream', location.href)
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(url)
    document.querySelectorAll('audio').forEach((element) => element.pause())
    const gain = audio.createGain()
    gain.gain.value = monitor ? 1 : 0
    gain.connect(audio.destination)
    const session: Session = {
      audio, socket, monitor: gain, pending: [], sent: [], awaiting: false, playAt: 0, inputLevel: 0,
      computeMs: 0, processedChunks: 0,
    }
    current.current = session
    socket.binaryType = 'arraybuffer'
    setError('')
    setMetrics({ input: 0, output: 0, delay: 0, processing: 0, rtf: 0 })
    setState('starting')
    onBusy(true)

    function send() {
      if (session.awaiting || !session.pending.length || socket.readyState !== WebSocket.OPEN) return
      const { pcm, captured } = session.pending.shift()!
      session.awaiting = true
      session.sent.push({ captured, remaining: CHUNK })
      socket.send(pcm)
    }

    async function capture() {
      await audio.audioWorklet.addModule('/mic-worklet.js')
      if (current.current !== session) return
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } })
      session.stream = stream
      if (current.current !== session) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }
      const node = new AudioWorkletNode(audio, 'voicefocus-mic', { channelCount: 1, channelCountMode: 'explicit' })
      session.capture = node
      node.port.onmessage = ({ data }: MessageEvent<ArrayBuffer>) => {
        if (current.current !== session) return
        if (session.pending.length >= MAX_PENDING || session.sent.length >= 40) {
          stop(session, '처리가 입력 속도를 따라가지 못해 마이크를 정지했습니다.')
          return
        }
        session.inputLevel = level(new Float32Array(data))
        session.pending.push({ pcm: data, captured: performance.now() - CHUNK / RATE * 1000 })
        send()
      }
      stream.getAudioTracks().forEach((track) => {
        track.onended = () => stop(session, '마이크 입력이 중단되었습니다.')
      })
      audio.createMediaStreamSource(stream).connect(node).connect(audio.destination)
      setState('live')
    }

    socket.onopen = () => socket.send(enrollment)
    socket.onclose = () => stop(session, '실시간 음성 연결이 종료되었습니다.')
    socket.onmessage = ({ data }: MessageEvent<string | ArrayBuffer>) => {
      if (current.current !== session) return
      if (typeof data === 'string') {
        const message = JSON.parse(data)
        if (message.error) stop(session, message.error)
        else if (message.ready) {
          window.clearTimeout(session.timeout)
          void capture().catch(() => stop(session, '마이크 권한과 오디오 장치를 확인해 주세요.'))
        } else if (typeof message.processing_ms === 'number') {
          session.awaiting = false
          session.computeMs += message.processing_ms
          session.processedChunks += 1
          const rtf = session.computeMs / (session.processedChunks * CHUNK / RATE * 1000)
          setMetrics((previous) => ({ ...previous, input: session.inputLevel, processing: message.processing_ms, rtf }))
          send()
        }
        return
      }
      const samples = new Float32Array(data)
      if (!samples.length) return
      const measured = session.sent[0]?.captured ?? performance.now()
      for (let remaining = samples.length; remaining > 0 && session.sent.length;) {
        const oldest = session.sent[0]
        const consumed = Math.min(remaining, oldest.remaining)
        oldest.remaining -= consumed
        oldest.captured += consumed / RATE * 1000
        remaining -= consumed
        if (oldest.remaining === 0) session.sent.shift()
      }
      const buffer = audio.createBuffer(1, samples.length, RATE)
      buffer.getChannelData(0).set(samples)
      const source = audio.createBufferSource()
      source.buffer = buffer
      source.connect(gain)
      source.onended = () => source.disconnect()
      if (session.playAt - audio.currentTime > 2) {
        stop(session, '재생 지연이 커져 마이크를 정지했습니다.')
        return
      }
      const startAt = Math.max(audio.currentTime + 0.04, session.playAt)
      source.start(startAt)
      session.playAt = startAt + buffer.duration
      const delay = Math.round(performance.now() - measured + (startAt - audio.currentTime) * 1000)
      setMetrics((previous) => ({ ...previous, delay, output: level(samples) }))
    }
    session.timeout = window.setTimeout(() => stop(session, '실시간 모델의 응답 시간이 초과되었습니다.'), 30_000)
    try {
      if (audio.sampleRate !== RATE) throw new Error('Unsupported sample rate')
      await audio.resume()
    } catch {
      stop(session, '브라우저에서 16 kHz 오디오를 시작할 수 없습니다.')
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
          <button type="button" className="secondary" onClick={() => { if (current.current) stop(current.current) }}>
            <Square size={15} aria-hidden="true" /> 마이크 정지
          </button>
        )}
      </div>
      {!available && <p>서버에 실시간 모델을 설치하면 사용할 수 있습니다.</p>}
      <label className="live-monitor"><input type="checkbox" checked={monitor} onChange={(event) => {
        const enabled = event.currentTarget.checked
        setMonitor(enabled)
        if (current.current) current.current.monitor.gain.value = enabled ? 1 : 0
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
