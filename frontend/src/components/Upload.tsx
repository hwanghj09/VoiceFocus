import { useId, useRef, useState } from 'react'
import { FileAudio, UploadCloud, X } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

const MAX_SIZE = 16 * 1024 * 1024

export function Upload({ label, hint, icon: Icon, file, disabled, onChange }: {
  label: string
  hint: string
  icon: LucideIcon
  file: File | null
  disabled: boolean
  onChange: (file: File | null) => void
}) {
  const inputId = useId()
  const hintId = useId()
  const input = useRef<HTMLInputElement>(null)
  const [dropping, setDropping] = useState(false)
  const [error, setError] = useState('')

  function choose(next?: File) {
    if (!next || disabled) return
    if (!next.name.toLowerCase().endsWith('.wav') || next.size > MAX_SIZE || !next.size) {
      setError('16 MiB 이하의 비어 있지 않은 WAV 파일을 선택해 주세요.')
      return
    }
    setError('')
    onChange(next)
  }

  return (
    <div className="upload">
      <h3><Icon aria-hidden="true" size={18} /><label htmlFor={inputId}>{label}</label></h3>
      <p id={hintId}>{hint}</p>
      <input
        className="visually-hidden" id={inputId} type="file" ref={input}
        accept=".wav,audio/wav,audio/x-wav" aria-describedby={hintId} disabled={disabled}
        onChange={(event) => {
          choose(event.currentTarget.files?.[0])
          event.currentTarget.value = ''
        }}
      />
      <div
        className={`dropzone ${dropping ? 'dragging' : ''} ${file ? 'selected' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setDropping(true)
        }}
        onDragLeave={() => setDropping(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDropping(false)
          choose(event.dataTransfer.files[0])
        }}
      >
        {file ? (
          <>
            <span className="drop-icon"><FileAudio aria-hidden="true" size={20} /></span>
            <span className="file-name" title={file.name}>{file.name}</span>
            <div className="file-actions">
              <button type="button" className="button-link" disabled={disabled} onClick={() => input.current?.click()}>변경</button>
              <button type="button" className="icon-button" disabled={disabled} aria-label={`${label} 지우기`} onClick={() => {
                setError('')
                onChange(null)
              }}><X size={15} aria-hidden="true" /></button>
            </div>
          </>
        ) : (
          <button type="button" className="pick-file" aria-label={`${label} 파일 선택`} disabled={disabled} onClick={() => input.current?.click()}>
            <span className="drop-icon"><UploadCloud aria-hidden="true" size={20} /></span>
            <strong>파일 선택 또는 끌어 놓기</strong>
            <small>WAV · 최대 16 MiB</small>
          </button>
        )}
      </div>
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  )
}
