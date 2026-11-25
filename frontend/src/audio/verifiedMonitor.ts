import { audioSocket } from './connection'

const RATE = 16000
const CHUNK = 2048
const MAX_PENDING = 8

export const INITIAL_METRICS = { input: 0, output: 0, delay: 0, processing: 0, rtf: 0 }

function level(samples: Float32Array) {
  let squares = 0
  for (const sample of samples) squares += sample * sample
  return Math.min(1, Math.sqrt(squares / samples.length) * 4)
}

export function startVerifiedMonitor(options: {
  enrollment: File
  input: () => Promise<MediaStream>
  ownsInput: boolean
  audible: boolean
  onReady: () => void
  onError: (message: string) => void
  onMetrics: (metrics: typeof INITIAL_METRICS) => void
}) {
  const audio = new AudioContext({ sampleRate: RATE })
  if (audio.sampleRate !== RATE) {
    void audio.close().catch(() => undefined)
    throw new Error('브라우저에서 16 kHz 오디오를 시작할 수 없습니다.')
  }
  const socket = audioSocket('/ws/v1/stream')
  const gain = audio.createGain()
  gain.gain.value = options.audible ? 1 : 0
  gain.connect(audio.destination)
  socket.binaryType = 'arraybuffer'

  const pending: { pcm: ArrayBuffer; captured: number }[] = []
  const sent: { remaining: number; captured: number }[] = []
  let stream: MediaStream | undefined
  let worklet: AudioWorkletNode | undefined
  let active = true
  let awaiting = false
  let playAt = 0
  let inputLevel = 0
  let computeMs = 0
  let processedChunks = 0
  let metrics = INITIAL_METRICS
  const timeout = window.setTimeout(() => fail('실시간 모델의 응답 시간이 초과되었습니다.'), 30_000)

  function close() {
    if (!active) return
    active = false
    window.clearTimeout(timeout)
    socket.onopen = null
    socket.onclose = null
    socket.onmessage = null
    socket.close()
    worklet?.disconnect()
    worklet?.port.close()
    if (options.ownsInput) stream?.getTracks().forEach((track) => track.stop())
    void audio.close().catch(() => undefined)
  }

  function fail(message: string) {
    if (!active) return
    close()
    options.onError(message)
  }

  function send() {
    if (awaiting || !pending.length || socket.readyState !== WebSocket.OPEN) return
    const { pcm, captured } = pending.shift()!
    awaiting = true
    sent.push({ captured, remaining: CHUNK })
    socket.send(pcm)
  }

  async function capture() {
    await audio.audioWorklet.addModule('/mic-worklet.js')
    if (!active) return
    stream = await options.input()
    if (!active) {
      if (options.ownsInput) stream.getTracks().forEach((track) => track.stop())
      return
    }
    worklet = new AudioWorkletNode(audio, 'voicefocus-mic', { channelCount: 1, channelCountMode: 'explicit' })
    worklet.port.onmessage = ({ data }: MessageEvent<ArrayBuffer>) => {
      if (!active) return
      if (pending.length >= MAX_PENDING || sent.length >= 40) {
        fail('처리가 입력 속도를 따라가지 못해 음성 처리가 정지됐습니다.')
        return
      }
      inputLevel = level(new Float32Array(data))
      pending.push({ pcm: data, captured: performance.now() - CHUNK / RATE * 1000 })
      send()
    }
    if (options.ownsInput) {
      stream.getAudioTracks().forEach((track) => { track.onended = () => fail('마이크 입력이 중단되었습니다.') })
    }
    audio.createMediaStreamSource(stream).connect(worklet).connect(audio.destination)
    options.onReady()
  }

  socket.onopen = () => socket.send(options.enrollment)
  socket.onclose = () => fail('실시간 음성 연결이 종료되었습니다.')
  socket.onmessage = ({ data }: MessageEvent<string | ArrayBuffer>) => {
    if (typeof data === 'string') {
      const message = JSON.parse(data)
      if (message.error) fail(message.error)
      else if (message.ready) {
        window.clearTimeout(timeout)
        void capture().catch(() => fail('오디오 입력을 시작할 수 없습니다.'))
      } else if (typeof message.processing_ms === 'number') {
        awaiting = false
        computeMs += message.processing_ms
        processedChunks += 1
        metrics = { ...metrics, input: inputLevel, processing: message.processing_ms, rtf: computeMs / (processedChunks * CHUNK / RATE * 1000) }
        options.onMetrics(metrics)
        send()
      }
      return
    }
    const samples = new Float32Array(data)
    if (!samples.length) return
    const measured = sent[0]?.captured ?? performance.now()
    for (let remaining = samples.length; remaining > 0 && sent.length;) {
      const oldest = sent[0]
      const consumed = Math.min(remaining, oldest.remaining)
      oldest.remaining -= consumed
      oldest.captured += consumed / RATE * 1000
      remaining -= consumed
      if (oldest.remaining === 0) sent.shift()
    }
    if (playAt - audio.currentTime > 2) {
      fail('재생 지연이 커져 음성 처리가 정지됐습니다.')
      return
    }
    const buffer = audio.createBuffer(1, samples.length, RATE)
    buffer.getChannelData(0).set(samples)
    const source = audio.createBufferSource()
    source.buffer = buffer
    source.connect(gain)
    source.onended = () => source.disconnect()
    const startAt = Math.max(audio.currentTime + 0.04, playAt)
    source.start(startAt)
    playAt = startAt + buffer.duration
    const delay = Math.round(performance.now() - measured + (startAt - audio.currentTime) * 1000)
    metrics = { ...metrics, delay, output: level(samples) }
    options.onMetrics(metrics)
  }
  void audio.resume().catch(() => fail('브라우저에서 오디오를 시작할 수 없습니다.'))
  return { close, setAudible: (audible: boolean) => { gain.gain.value = audible ? 1 : 0 } }
}
