import { useEffect, useState } from 'react'

export function clock(seconds: number): string {
  const value = Math.floor(seconds)
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`
}

export function useBlobUrl(blob: Blob | null): string {
  const [url, setUrl] = useState('')
  useEffect(() => {
    if (!blob) {
      setUrl('')
      return
    }
    const next = URL.createObjectURL(blob)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [blob])
  return url
}
