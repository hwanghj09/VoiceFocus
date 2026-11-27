export interface ModelInfo {
  id: string
  name: string
  version: string
  sample_rate: number
  available: boolean
}

export interface ModelStatus {
  models: ModelInfo[]
  example_available: boolean
  automatic_available: boolean
  live_available: boolean
}

export type Mode = 'speech' | 'call'

export interface Job {
  id: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  error: string | null
  processing_ms: number | null
  inference_ms: number | null
  duration_seconds: number
  mode: Mode
  model_version: string
  target_muted: boolean | null
}

const BASE = '/api/v1'

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(BASE + path, { ...init, cache: 'no-store' })
  if (!response.ok) {
    const error = await response.json().catch(() => undefined)
    throw new Error(typeof error?.detail === 'string' ? error.detail : '서버 요청을 완료하지 못했습니다.')
  }
  return response
}

export async function getModels(signal: AbortSignal): Promise<ModelStatus> {
  return (await request('/models', { signal })).json()
}

export async function getExample(name: string, signal: AbortSignal): Promise<File> {
  const response = await request('/examples/' + name, { signal })
  return new File([await response.blob()], name, { type: 'audio/wav' })
}

export async function extract(
  mixture: File, enrollment: File, signal: AbortSignal,
): Promise<Job> {
  const body = new FormData()
  body.append('mixture', mixture)
  body.append('enrollment', enrollment)
  return (await request('/extract?mode=call', { method: 'POST', body, signal })).json()
}

export async function mix(a: File, b: File, signal: AbortSignal): Promise<File> {
  const body = new FormData()
  body.append('a', a)
  body.append('b', b)
  const response = await request('/mix', { method: 'POST', body, signal })
  return new File([await response.blob()], 'voicefocus-mixed.wav', { type: 'audio/wav' })
}

export async function getJob(id: string, signal: AbortSignal): Promise<Job> {
  return (await request('/jobs/' + id, { signal })).json()
}

export async function deleteJob(id: string, signal: AbortSignal): Promise<void> {
  await request('/jobs/' + id, { method: 'DELETE', signal })
}

export async function getAudio(id: string, type: 'original' | 'result', signal: AbortSignal): Promise<Blob> {
  return (await request('/jobs/' + id + '/' + type, { signal })).blob()
}

export function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted()
    const abort = () => {
      window.clearTimeout(timeout)
      reject(signal.reason)
    }
    const timeout = window.setTimeout(() => {
      signal.removeEventListener('abort', abort)
      resolve()
    }, ms)
    signal.addEventListener('abort', abort, { once: true })
  })
}
