import { useEffect, useRef, useState } from 'react'
import { AudioLines, Download, Pause, Play } from 'lucide-react'
import { clock, useBlobUrl } from '../audio/playback'

const RATE = 16_000
const COLOR = { input: '#4a58bf', output: '#147e76', residual: '#9c4c1d' }

function Player({ url, label }: { url: string; label: string }) {
  const audio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [position, setPosition] = useState(0)
  const [error, setError] = useState('')

  async function toggle() {
    const element = audio.current
    if (!element) return
    if (!element.paused) {
      element.pause()
      return
    }
    for (const other of document.querySelectorAll('audio')) {
      if (other !== element) other.pause()
    }
    try {
      await element.play()
    } catch {
      setError('이 오디오를 재생할 수 없습니다.')
    }
  }

  return (
    <>
      <div className="player" aria-label={label + ' 재생기'}>
        <audio
          ref={audio} src={url} preload="metadata"
          onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
          onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
          onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
          onError={() => setError('오디오를 읽을 수 없습니다.')}
        />
        <button type="button" className="play" onClick={toggle} aria-label={label + (playing ? ' 일시정지' : ' 재생')}>
          {playing ? <Pause size={16} fill="currentColor" aria-hidden="true" /> : <Play size={16} fill="currentColor" aria-hidden="true" />}
        </button>
        <span className="time">{clock(position)}</span>
        <input
          aria-label={label + ' 재생 위치'} type="range" min={0} max={duration || 1} step={0.01}
          value={Math.min(position, duration || 1)} disabled={!duration}
          onChange={(event) => {
            const next = Number(event.currentTarget.value)
            if (audio.current) audio.current.currentTime = next
            setPosition(next)
          }}
        />
        <span className="time">{clock(duration)}</span>
      </div>
      {error && <p className="field-error" role="alert">{error}</p>}
    </>
  )
}

function drawWave(canvas: HTMLCanvasElement, samples: Float32Array, accent: string) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const { width, height } = canvas
  ctx.clearRect(0, 0, width, height)
  ctx.beginPath()
  for (let x = 0; x < width; x++) {
    const start = Math.floor(x * samples.length / width)
    const end = Math.max(start + 1, Math.floor((x + 1) * samples.length / width))
    let low = 0
    let high = 0
    for (let i = start; i < end; i++) {
      low = Math.min(low, samples[i])
      high = Math.max(high, samples[i])
    }
    ctx.moveTo(x, (1 - high) * height / 2)
    ctx.lineTo(x, (1 - low) * height / 2 + 0.6)
  }
  ctx.strokeStyle = accent
  ctx.stroke()
}

function AudioPlots({ blob, label, accent, progress }: { blob: Blob; label: string; accent: string; progress?: number }) {
  const wave = useRef<HTMLCanvasElement>(null)
  const spectrum = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState('')
  const [duration, setDuration] = useState<number | null>(null)
  useEffect(() => {
    let active = true
    let worker: Worker | undefined
    async function plot() {
      try {
        const decoder = new OfflineAudioContext(1, 1, RATE)
        const buffer = await decoder.decodeAudioData(await blob.arrayBuffer())
        if (!active || !wave.current) return
        const samples = buffer.getChannelData(0).slice()
        drawWave(wave.current, samples, accent)
        setDuration(buffer.duration)
        worker = new Worker(new URL('../spectrum.worker.ts', import.meta.url), { type: 'module' })
        worker.onmessage = (event: MessageEvent<{ width: number; height: number; pixels: Uint8ClampedArray<ArrayBuffer> }>) => {
          const canvas = spectrum.current
          if (!active || !canvas) return
          const { width, height, pixels } = event.data
          canvas.width = width
          canvas.height = height
          canvas.getContext('2d')?.putImageData(new ImageData(pixels, width, height), 0, 0)
          worker?.terminate()
        }
        worker.onerror = () => {
          if (active) setError('스펙트로그램을 생성할 수 없습니다.')
          worker?.terminate()
        }
        worker.postMessage(samples, [samples.buffer])
      } catch {
        if (active) setError('오디오 시각화를 생성할 수 없습니다.')
      }
    }
    void plot()
    return () => {
      active = false
      worker?.terminate()
    }
  }, [blob, accent])
  const marker = progress === undefined ? null : (
    <span className="plot-playhead" aria-hidden="true" style={{ left: `${progress * 100}%` }} />
  )
  return (
    <>
      <div className="plot-title"><span>파형</span><span>진폭 −1 ~ +1</span></div>
      <div className="plot">
        <canvas className="wave" ref={wave} width={1024} height={184} role="img" aria-label={label + ' 파형'} />
        {marker}
      </div>
      <div className="plot-title"><span>스펙트로그램</span><span>0–8 kHz · −85 ~ −10 dBFS</span></div>
      <div className="plot">
        <canvas className="spectrum" ref={spectrum} role="img" aria-label={label + ' 스펙트로그램'} />
        {marker}
      </div>
      <div className="time-axis"><span>0초</span><span>{duration === null ? '시간' : `${duration.toFixed(2)}초`}</span></div>
      {error && <p className="field-error" role="alert">{error}</p>}
    </>
  )
}

export function AudioPreview({ label, description, tone, blob, download, filename, progress }: {
  label: string
  description: string
  tone: keyof typeof COLOR
  blob: Blob | null
  download?: Blob | null
  filename: string
  progress?: number
}) {
  const url = useBlobUrl(blob)
  const alternateUrl = useBlobUrl(download ?? null)
  return (
    <article className={`preview ${tone}`}>
      <div className="preview-header">
        <span className="preview-icon"><AudioLines size={19} aria-hidden="true" /></span>
        <div><h3>{label}</h3><p>{description}</p></div>
        {url && <a className="icon-button download" href={alternateUrl || url} download={filename} aria-label={label + ' WAV 다운로드'} title="WAV 다운로드"><Download size={18} aria-hidden="true" /></a>}
      </div>
      {url && blob ? (
        <>
          {progress === undefined && <Player key={url} url={url} label={label} />}
          <AudioPlots key={url + '-plots'} blob={blob} label={label} accent={COLOR[tone]} progress={progress} />
        </>
      ) : (
        <div className="empty-wave"><AudioLines size={23} aria-hidden="true" /><span>오디오를 기다리는 중</span></div>
      )}
    </article>
  )
}
