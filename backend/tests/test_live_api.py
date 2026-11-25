import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app import live, main


def test_live_stream_validates_input_and_releases_session(monkeypatch):
    monkeypatch.setattr(main, "checkpoint_exists", lambda _directory: True)

    class FakeSession:
        def push(self, samples):
            return samples * 0.5

    def prepare(_directory, _device, enrollment):
        assert len(enrollment) == 16000
        return FakeSession()

    monkeypatch.setattr(live, "load_stream", prepare)
    cue = io.BytesIO()
    sf.write(cue, np.full(16000, 0.1, dtype=np.float32), 16000, format="WAV")
    chunk = np.full(live.CHUNK, 0.2, dtype="<f4")

    with TestClient(main.create_app()) as client:
        with pytest.raises(WebSocketDisconnect) as blocked:
            with client.websocket_connect("/ws/v1/stream", headers={"origin": "https://other.example"}):
                pass
        assert blocked.value.code == 1008
        with client.websocket_connect("/ws/v1/stream") as socket:
            socket.send_bytes(cue.getvalue())
            assert socket.receive_json() == {"ready": True, "sample_rate": 16000, "chunk": live.CHUNK}
            with client.websocket_connect("/ws/v1/stream") as other:
                assert "실행 중" in other.receive_json()["error"]
            socket.send_bytes(chunk.tobytes())
            np.testing.assert_array_equal(np.frombuffer(socket.receive_bytes(), dtype="<f4"), 0.5 * chunk)
            assert socket.receive_json()["processing_ms"] >= 0
            socket.send_bytes(b"invalid")
            assert "형식" in socket.receive_json()["error"]
        with client.websocket_connect("/ws/v1/stream") as next_socket:
            next_socket.send_bytes(cue.getvalue())
            assert next_socket.receive_json()["ready"] is True
            with client.websocket_connect("/ws/v1/stream") as queued:
                queued.send_bytes(cue.getvalue())
                next_socket.close()
                assert queued.receive_json()["ready"] is True
