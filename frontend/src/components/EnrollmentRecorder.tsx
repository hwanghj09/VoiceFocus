import { useEffect, useRef, useState } from 'react'
import { Mic, Square, X } from 'lucide-react'
import { enrollmentWav, MAX_RECORDING_SECONDS } from '../audio/recording'

type Status = 'idle' | 'requesting' | 'recording' | 'encoding'

interface Recording {
  stream?: MediaStream
  recorder?: MediaRecorder
  timer?: number
  interval?: number
}

function stopCapture(capture: Recording) {
  window.clearTimeout(capture.timer)
  window.clearInterval(capture.interval)
  if (capture.recorder && capture.recorder.state !== 'inactive') capture.recorder.stop()
  capture.stream?.getTracks().forEach((track) => track.stop())
}

function recordingError(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') return '브라우저에서 마이크 권한을 허용해 주세요.'
    if (error.name === 'NotFoundError') return '사용할 수 있는 마이크가 없습니다.'
    return '마이크 음성을 녹음하거나 읽을 수 없습니다.'
  }
  return error instanceof Error ? error.message : '녹음 중 오류가 발생했습니다.'
}

export function EnrollmentRecorder({ active, disabled, onBusy, onRecorded, label = '등록 음성 녹음', maxSeconds = MAX_RECORDING_SECONDS, filename = 'voicefocus-enrollment.wav' }: {
  active: boolean
  label?: string
  maxSeconds?: number
  filename?: string
  disabled: boolean
  onBusy: (busy: boolean) => void
  onRecorded: (file: File) => void
}) {
  const current = useRef<Recording | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!active) {
      setStatus('idle')
      setError('')
      onBusy(false)
    }
    return () => {
      const capture = current.current
      current.current = null
      if (capture) stopCapture(capture)
    }
  }, [active, onBusy])

  function reset(capture: Recording) {
    if (current.current !== capture) return
    current.current = null
    stopCapture(capture)
    setStatus('idle')
    onBusy(false)
  }

  function stop(capture: Recording) {
    if (current.current !== capture) return
    setStatus('encoding')
    stopCapture(capture)
  }

  async function start() {
    if (!active || current.current || disabled) return
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('이 브라우저에서는 마이크 녹음을 사용할 수 없습니다.')
      return
    }
    const capture: Recording = {}
    current.current = capture
    setError('')
    setSeconds(0)
    setStatus('requesting')
    onBusy(true)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      capture.stream = stream
      if (current.current !== capture) {
        stopCapture(capture)
        return
      }
      const recorder = new MediaRecorder(stream)
      const chunks: Blob[] = []
      capture.recorder = recorder
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data)
      }
      recorder.onerror = () => {
        if (current.current !== capture) return
        setError('마이크 음성을 녹음할 수 없습니다.')
        reset(capture)
      }
      recorder.onstop = async () => {
        if (current.current !== capture) return
        stopCapture(capture)
        setStatus('encoding')
        try {
          const clip = new Blob(chunks, { type: recorder.mimeType })
          const file = await enrollmentWav(clip, maxSeconds, filename)
          if (current.current === capture) onRecorded(file)
        } catch (reason) {
          if (current.current === capture) setError(recordingError(reason))
        } finally {
          reset(capture)
        }
      }

      recorder.start()
      setStatus('recording')
      const started = performance.now()
      capture.interval = window.setInterval(() => {
        setSeconds(Math.min(maxSeconds, Math.floor((performance.now() - started) / 1000)))
      }, 200)
      capture.timer = window.setTimeout(() => stop(capture), maxSeconds * 1000)
    } catch (reason) {
      if (current.current !== capture) return
      setError(recordingError(reason))
      reset(capture)
    }
  }

  return (
    <div className="enrollment-recorder">
      {status === 'idle' && (
        <button type="button" className="secondary" disabled={disabled} onClick={() => void start()}>
          <Mic size={15} aria-hidden="true" /> {label}
        </button>
      )}
      {status === 'requesting' && (
        <>
          <span role="status">마이크 권한 확인 중</span>
          <button type="button" className="icon-button" aria-label="녹음 취소" onClick={() => { if (current.current) reset(current.current) }}>
            <X size={16} aria-hidden="true" />
          </button>
        </>
      )}
      {status === 'recording' && (
        <>
          <button type="button" className="secondary" onClick={() => { if (current.current) stop(current.current) }}>
            <Square size={14} aria-hidden="true" /> 녹음 종료
          </button>
          <span>녹음 중 · {seconds} / {maxSeconds}초</span>
        </>
      )}
      {status === 'encoding' && <span role="status">WAV 변환 중</span>}
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  )
}
