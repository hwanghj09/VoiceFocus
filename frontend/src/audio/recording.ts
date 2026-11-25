const RATE = 16_000
export const MAX_RECORDING_SECONDS = 20

export async function enrollmentWav(recording: Blob): Promise<File> {
  const decoder = new OfflineAudioContext(1, 1, RATE)
  const decoded = await decoder.decodeAudioData(await recording.arrayBuffer())
  const length = Math.min(decoded.length, RATE * MAX_RECORDING_SECONDS)
  if (length < RATE) throw new Error('등록 음성은 1초 이상 녹음해 주세요.')

  const buffer = new ArrayBuffer(44 + 2 * length)
  const wav = new DataView(buffer)
  for (const [offset, value] of [[0, 'RIFF'], [8, 'WAVE'], [12, 'fmt '], [36, 'data']] as const) {
    for (let i = 0; i < value.length; i++) wav.setUint8(offset + i, value.charCodeAt(i))
  }
  wav.setUint32(4, buffer.byteLength - 8, true)
  wav.setUint32(16, 16, true)
  wav.setUint16(20, 1, true)
  wav.setUint16(22, 1, true)
  wav.setUint32(24, RATE, true)
  wav.setUint32(28, RATE * 2, true)
  wav.setUint16(32, 2, true)
  wav.setUint16(34, 16, true)
  wav.setUint32(40, 2 * length, true)

  const channels = Array.from({ length: decoded.numberOfChannels }, (_, index) => decoded.getChannelData(index))
  for (let i = 0; i < length; i++) {
    let sum = 0
    for (const channel of channels) sum += channel[i]
    const sample = Math.max(-1, Math.min(1, sum / channels.length))
    wav.setInt16(44 + 2 * i, Math.round(sample * (sample < 0 ? 32768 : 32767)), true)
  }
  return new File([buffer], 'voicefocus-enrollment.wav', { type: 'audio/wav' })
}
