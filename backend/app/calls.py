"""In-memory signaling for two-browser WebRTC experiments."""

import asyncio
import json
import re
from dataclasses import dataclass
from secrets import token_hex

from fastapi import WebSocket, WebSocketDisconnect

from backend.app.live import is_same_origin

MAX_ROOMS = 32
MAX_MESSAGE = 24_576
ROOM_CODE = re.compile(r"[0-9A-F]{12}")


@dataclass
class Room:
    host: WebSocket
    guest: WebSocket | None = None
    offered: bool = False
    answered: bool = False


async def receive(socket: WebSocket) -> dict:
    packet = await socket.receive()
    if packet["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(packet["code"])
    data = packet.get("text")
    if data is None or len(data) > MAX_MESSAGE:
        raise ValueError("통화 연결 메시지를 읽을 수 없습니다.")
    try:
        message = json.loads(data)
    except ValueError:
        raise ValueError("통화 연결 메시지를 읽을 수 없습니다.") from None
    if not isinstance(message, dict):
        raise ValueError("통화 연결 메시지를 읽을 수 없습니다.")
    return message


async def notify(socket: WebSocket, message: dict) -> None:
    try:
        await socket.send_json(message)
    except (WebSocketDisconnect, RuntimeError):
        pass  # The peer may have closed its socket at the same time.


async def serve_call(socket: WebSocket, rooms: dict[str, Room]) -> None:
    if not is_same_origin(socket):
        await socket.close(code=1008)
        return
    await socket.accept()
    room = None
    code = ""
    try:
        request = await asyncio.wait_for(receive(socket), 10)
        action = request.get("type")
        if action == "create":
            if len(rooms) >= MAX_ROOMS:
                raise ValueError("현재 통화방이 가득 찼습니다.")
            code = token_hex(6).upper()
            while code in rooms:
                code = token_hex(6).upper()
            room = rooms[code] = Room(socket)
        elif action == "join":
            code = request.get("code")
            if not isinstance(code, str) or not ROOM_CODE.fullmatch(code) or code not in rooms:
                raise ValueError("사용할 수 있는 방 코드를 입력해 주세요.")
            existing = rooms[code]
            if existing.guest is not None:
                raise ValueError("이미 두 사람이 참가한 방입니다.")
            room = existing
            room.guest = socket
        else:
            raise ValueError("방을 만들거나 방 코드로 참가해 주세요.")

        await socket.send_json({"type": "joined", "code": code})
        if socket is room.guest:
            await notify(room.host, {"type": "peer_ready"})

        while rooms.get(code) is room:
            message = await receive(socket)
            if rooms.get(code) is not room:
                break
            kind = message.get("type")
            if kind == "leave":
                await socket.close()
                return
            sdp = message.get("sdp")
            if not isinstance(sdp, str) or not sdp.startswith("v=0"):
                raise ValueError("통화 연결 정보가 올바르지 않습니다.")
            if kind == "offer" and socket is room.host and room.guest and not room.offered:
                room.offered = True
                await notify(room.guest, {"type": "offer", "sdp": sdp})
            elif kind == "answer" and socket is room.guest and room.offered and not room.answered:
                room.answered = True
                await notify(room.host, {"type": "answer", "sdp": sdp})
            else:
                raise ValueError("통화 연결 순서가 올바르지 않습니다.")
        await socket.close()
    except WebSocketDisconnect:
        pass
    except (ValueError, TimeoutError) as error:
        await notify(socket, {"type": "error", "message": str(error) or "통화방 연결 시간이 초과되었습니다."})
        try:
            await socket.close(code=1008)
        except (WebSocketDisconnect, RuntimeError):
            pass
    finally:
        if room is not None and rooms.get(code) is room:
            del rooms[code]
            other = room.guest if socket is room.host else room.host
            if other is not None:
                await notify(other, {"type": "ended"})
