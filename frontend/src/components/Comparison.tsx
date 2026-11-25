import { useEffect, useRef, useState } from 'react'
import { Pause, Play } from 'lucide-react'
import { clock, useBlobUrl } from '../audio/playback'
import { AudioPreview } from './AudioPreview'

const TRACKS = ['original', 'output'] as const
const TRACK_LABEL = { original: '원본 혼합 음성', output: '추정 음성' }
type Track = (typeof TRACKS)[number]

export function Comparison({ original, output, inputFile, duration }: {
  original: Blob
  output: Blob
  inputFile: File | null
  duration: number
}) {
  const sources = { original: useBlobUrl(original), output: useBlobUrl(output) }
  const media = useRef<Partial<Record<Track, HTMLAudioElement>>>({})
  const active = useRef<Track>('original')
  const [track, setTrack] = useState<Track>('original')
  const [loaded, setLoaded] = useState<Partial<Record<Track, boolean>>>({})
  const [position, setPosition] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [error, setError] = useState('')
  const ready = loaded.original && loaded.output
  const progress = duration > 0 ? Math.min(position / duration, 1) : 0

  useEffect(() => {
    const elements = Object.values(media.current)
    return () => elements.forEach((element) => element.pause())
  }, [])

  async function play(element: HTMLAudioElement) {
    setError('')
    try {
      await element.play()
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') return
      if (media.current[active.current] === element) {
        setPlaying(false)
        setError('선택한 음성을 재생할 수 없습니다.')
      }
    }
  }

  function togglePlayback() {
    const element = media.current[active.current]
    if (!ready || !element) return
    if (element.paused) void play(element)
    else element.pause()
  }

  function selectTrack(next: Track) {
    if (!ready || next === active.current) return
    const from = media.current[active.current]
    const to = media.current[next]
    if (!from || !to) return
    const continuing = !from.paused
    const time = Math.min(from.currentTime, to.duration)
    active.current = next
    from.pause()
    to.currentTime = time
    setTrack(next)
    setPosition(time)
    setPlaying(continuing)
    if (continuing) void play(to)
    else setError('')
  }

  return (
    <>
      <div className="compare-player">
        {TRACKS.map((source) => (
          <audio
            key={source}
            ref={(element) => {
              if (element) media.current[source] = element
              else delete media.current[source]
            }}
            src={sources[source] || undefined}
            preload="auto"
            onLoadedMetadata={() => setLoaded((previous) => ({ ...previous, [source]: true }))}
            onTimeUpdate={(event) => {
              if (active.current === source) setPosition(event.currentTarget.currentTime)
            }}
            onPlay={() => {
              if (active.current === source) setPlaying(true)
            }}
            onPause={() => {
              if (active.current === source) setPlaying(false)
            }}
            onEnded={() => {
              if (active.current === source) setPlaying(false)
            }}
            onError={() => {
              setLoaded((previous) => ({ ...previous, [source]: false }))
              setError(`${TRACK_LABEL[source]}을 읽을 수 없습니다.`)
            }}
          />
        ))}
        <div className="compare-tracks" role="group" aria-label="비교 음성 선택">
          {TRACKS.map((source) => (
            <button
              key={source} type="button" aria-pressed={track === source} disabled={!ready}
              onClick={() => selectTrack(source)}
            >
              <span aria-hidden="true" className={`track-dot ${source}`} />
              {TRACK_LABEL[source]}
            </button>
          ))}
        </div>
        <div className="player compare-timeline" aria-label="공통 재생기">
          <button type="button" className="play" disabled={!ready} aria-label={playing ? '비교 재생 일시정지' : '비교 재생'} onClick={togglePlayback}>
            {playing ? <Pause size={16} fill="currentColor" aria-hidden="true" /> : <Play size={16} fill="currentColor" aria-hidden="true" />}
          </button>
          <span className="time">{clock(position)}</span>
          <input
            type="range" min={0} max={duration || 1} step={0.01} value={Math.min(position, duration || 1)}
            disabled={!ready} aria-label="공통 재생 위치" aria-valuetext={clock(position)}
            onChange={(event) => {
              const next = Number(event.currentTarget.value)
              const element = media.current[active.current]
              if (element) element.currentTime = next
              setPosition(next)
            }}
          />
          <span className="time">{clock(duration)}</span>
        </div>
        {error && <p className="field-error" role="alert">{error}</p>}
      </div>
      <div className="preview-grid">
        <AudioPreview tone="input" label={TRACK_LABEL.original} description="비교 재생용 · 16 kHz 모노" blob={original} download={inputFile} filename={inputFile?.name || 'voicefocus-original.wav'} progress={progress} />
        <AudioPreview tone="output" label={TRACK_LABEL.output} description="등록한 목표 화자의 추출 결과" blob={output} filename="voicefocus-extracted.wav" progress={progress} />
      </div>
    </>
  )
}
