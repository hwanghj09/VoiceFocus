import { audioSocket } from './connection'

export type CallStatus = 'idle' | 'starting' | 'waiting' | 'connecting' | 'connected'

type Action = { type: 'create' } | { type: 'join'; code: string }

function gatherIce(peer: RTCPeerConnection, signal: AbortSignal) {
  if (peer.iceGatheringState === 'complete' || signal.aborted) return Promise.resolve()
  return new Promise<void>((resolve) => {
    const timer = window.setTimeout(finish, 5000)
    function finish() {
      window.clearTimeout(timer)
      peer.removeEventListener('icegatheringstatechange', changed)
      signal.removeEventListener('abort', finish)
      resolve()
    }
    function changed() {
      if (peer.iceGatheringState === 'complete') finish()
    }
    peer.addEventListener('icegatheringstatechange', changed)
    signal.addEventListener('abort', finish, { once: true })
  })
}

export function startPeerCall(action: Action, callbacks: {
  onRoom: (code: string) => void
  onStatus: (state: CallStatus) => void
  onRemote: (stream: MediaStream) => void
  onEnd: (message: string) => void
}) {
  const peer = new RTCPeerConnection()
  const controller = new AbortController()
  let microphone: MediaStream | undefined
  let socket: WebSocket | undefined
  let timeout: number | undefined

  function close() {
    if (controller.signal.aborted) return
    controller.abort()
    window.clearTimeout(timeout)
    if (socket) {
      socket.onopen = null
      socket.onmessage = null
      socket.onclose = null
      socket.close()
    }
    peer.close()
    microphone?.getTracks().forEach((track) => track.stop())
  }

  function end(message: string) {
    if (controller.signal.aborted) return
    close()
    callbacks.onEnd(message)
  }

  function connect() {
    callbacks.onStatus('connecting')
    window.clearTimeout(timeout)
    timeout = window.setTimeout(() => end('상대 브라우저에 연결할 수 없습니다. 같은 네트워크인지 확인해 주세요.'), 20_000)
  }

  peer.ontrack = (event) => {
    if (!controller.signal.aborted) callbacks.onRemote(event.streams[0] ?? new MediaStream([event.track]))
  }
  peer.onconnectionstatechange = () => {
    if (controller.signal.aborted) return
    if (peer.connectionState === 'connected') {
      window.clearTimeout(timeout)
      callbacks.onStatus('connected')
    } else if (peer.connectionState === 'disconnected') {
      connect()
    } else if (peer.connectionState === 'failed') {
      end('상대 브라우저에 연결할 수 없습니다. 같은 네트워크인지 확인해 주세요.')
    }
  }

  async function localDescription(type: 'offer' | 'answer') {
    const description = type === 'offer' ? await peer.createOffer() : await peer.createAnswer()
    await peer.setLocalDescription(description)
    await gatherIce(peer, controller.signal)
    if (socket?.readyState === WebSocket.OPEN && !controller.signal.aborted) {
      socket.send(JSON.stringify({ type, sdp: peer.localDescription?.sdp }))
    }
  }

  async function receive(data: string) {
    const message = JSON.parse(data)
    if (message.type === 'error') {
      end(message.message)
    } else if (message.type === 'ended') {
      end('상대가 통화를 종료했습니다.')
    } else if (message.type === 'joined') {
      window.clearTimeout(timeout)
      callbacks.onRoom(message.code)
      if (action.type === 'create') callbacks.onStatus('waiting')
      else connect()
    } else if (message.type === 'peer_ready') {
      connect()
      await localDescription('offer')
    } else if (message.type === 'offer') {
      await peer.setRemoteDescription({ type: 'offer', sdp: message.sdp })
      await localDescription('answer')
    } else if (message.type === 'answer') {
      await peer.setRemoteDescription({ type: 'answer', sdp: message.sdp })
    }
  }

  async function open() {
    try {
      microphone = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true } })
      if (controller.signal.aborted) {
        microphone.getTracks().forEach((track) => track.stop())
        return
      }
      microphone.getAudioTracks().forEach((track) => {
        peer.addTrack(track, microphone!)
        track.onended = () => end('마이크 입력이 중단되었습니다.')
      })
      socket = audioSocket('/ws/v1/calls')
      timeout = window.setTimeout(() => end('통화 서버의 응답 시간이 초과되었습니다.'), 15_000)
      const channel = socket
      channel.onopen = () => channel.send(JSON.stringify(action))
      channel.onclose = () => end('통화 연결이 종료되었습니다.')
      channel.onmessage = ({ data }) => {
        if (controller.signal.aborted) return
        void receive(data).catch(() => end('상대 브라우저와 음성 연결을 설정할 수 없습니다.'))
      }
    } catch (error) {
      end(error instanceof DOMException && error.name === 'NotAllowedError'
        ? '브라우저에서 마이크 권한을 허용해 주세요.'
        : '마이크 또는 통화 연결을 시작할 수 없습니다.')
    }
  }

  void open()
  return { close }
}
