export function audioSocket(path: string) {
  const url = new URL(path, location.href)
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return new WebSocket(url)
}
