import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.main import create_app


def test_call_room_relays_descriptions_and_expires_after_leave():
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/v1/calls") as host:
            host.send_json({"type": "create"})
            joined = host.receive_json()
            assert joined["type"] == "joined"
            code = joined["code"]
            with client.websocket_connect("/ws/v1/calls") as guest:
                guest.send_json({"type": "join", "code": code})
                assert guest.receive_json() == joined
                assert host.receive_json()["type"] == "peer_ready"
                with client.websocket_connect("/ws/v1/calls") as third:
                    third.send_json({"type": "join", "code": code})
                    assert third.receive_json()["type"] == "error"
                offer = {"type": "offer", "sdp": "v=0\r\no=offer"}
                answer = {"type": "answer", "sdp": "v=0\r\no=answer"}
                host.send_json(offer)
                assert guest.receive_json() == offer
                guest.send_json(answer)
                assert host.receive_json() == answer
                guest.send_json({"type": "leave"})
                assert host.receive_json()["type"] == "ended"
            with client.websocket_connect("/ws/v1/calls") as late:
                late.send_json({"type": "join", "code": code})
                assert late.receive_json()["type"] == "error"


def test_calls_reject_cross_origin_connections_and_malformed_messages():
    with TestClient(create_app()) as client:
        with pytest.raises(WebSocketDisconnect) as blocked:
            with client.websocket_connect("/ws/v1/calls", headers={"origin": "https://other.example"}):
                pass
        assert blocked.value.code == 1008
        with client.websocket_connect("/ws/v1/calls") as socket:
            socket.send_json(["create"])
            assert socket.receive_json()["type"] == "error"
