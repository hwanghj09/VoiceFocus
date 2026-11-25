class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super()
    this.chunk = new Float32Array(2048)
    this.offset = 0
  }

  process(inputs) {
    const channel = inputs[0]?.[0]
    if (!channel) return true
    for (const sample of channel) {
      this.chunk[this.offset++] = Math.max(-1, Math.min(1, sample))
      if (this.offset === this.chunk.length) {
        this.port.postMessage(this.chunk.buffer, [this.chunk.buffer])
        this.chunk = new Float32Array(2048)
        this.offset = 0
      }
    }
    return true
  }
}

registerProcessor('voicefocus-mic', MicCapture)
