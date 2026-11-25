const SIZE = 512
const HALF = SIZE / 2
const indices = new Uint16Array(SIZE)
const windowing = new Float32Array(SIZE)
const cosines = new Float32Array(HALF)
const sines = new Float32Array(HALF)

for (let index = 0; index < SIZE; index++) {
  let reverse = 0
  for (let bit = 1; bit < SIZE; bit *= 2) {
    reverse = reverse * 2 + Number(Boolean(index & bit))
  }
  indices[index] = reverse
  windowing[index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / (SIZE - 1)))
}
for (let index = 0; index < HALF; index++) {
  cosines[index] = Math.cos((-2 * Math.PI * index) / SIZE)
  sines[index] = Math.sin((-2 * Math.PI * index) / SIZE)
}

self.onmessage = ({ data }: MessageEvent<Float32Array>) => {
  const width = Math.min(560, Math.ceil(data.length / 128))
  const real = new Float32Array(SIZE)
  const imaginary = new Float32Array(SIZE)
  const pixels = new Uint8ClampedArray(width * HALF * 4)
  for (let x = 0; x < width; x++) {
    const start = Math.round(x * (data.length - 1) / Math.max(1, width - 1)) - HALF
    for (let i = 0; i < SIZE; i++) {
      real[indices[i]] = windowing[i] * (data[start + i] ?? 0)
    }
    imaginary.fill(0)
    for (let length = 2; length <= SIZE; length *= 2) {
      const half = length / 2
      for (let block = 0; block < SIZE; block += length) {
        for (let j = 0; j < half; j++) {
          const phase = (j * SIZE) / length
          const a = block + j
          const b = a + half
          const r = real[b] * cosines[phase] - imaginary[b] * sines[phase]
          const i = real[b] * sines[phase] + imaginary[b] * cosines[phase]
          real[b] = real[a] - r
          imaginary[b] = imaginary[a] - i
          real[a] += r
          imaginary[a] += i
        }
      }
    }
    for (let bin = 0; bin < HALF; bin++) {
      const db = 20 * Math.log10(Math.hypot(real[bin], imaginary[bin]) / (SIZE / 4) + 1e-8)
      const intensity = Math.min(1, Math.max(0, (db + 85) / 75))
      const offset = ((HALF - bin - 1) * width + x) * 4
      pixels[offset] = 15 + intensity * 235
      pixels[offset + 1] = 21 + intensity ** 1.7 * 171
      pixels[offset + 2] = 39 + intensity * 102
      pixels[offset + 3] = 255
    }
  }
  self.postMessage({ pixels, width, height: HALF }, { transfer: [pixels.buffer] })
}
