import numpy as np
import pytest

from backend.app.models.realtse import DEFAULT_MODEL_DIR, VENDOR_DIR, checkpoint_exists


def test_causal_stream_preserves_samples_and_isolates_sessions():
    model = DEFAULT_MODEL_DIR.with_name("spk_emb_causal_100")
    if not checkpoint_exists(model) or not all((VENDOR_DIR / name).is_dir() for name in ("wesep-real-tse", "wespeaker")):
        pytest.skip("Install the causal checkpoint with scripts/setup_model.sh spk_emb_causal_100.")
    torch = pytest.importorskip("torch")

    from backend.app.models.causal import CausalStream
    from backend.app.models.worker import Extractor

    rng = np.random.default_rng(4)
    mix_a = rng.normal(0, 0.05, size=2103).astype(np.float32)
    mix_b = rng.normal(0, 0.04, size=1951).astype(np.float32)
    cue_a, cue_b = rng.normal(0, 0.04, size=(2, 16531)).astype(np.float32)

    extractor = Extractor(model)
    torch.manual_seed(42)
    offline_a = extractor.extract(mix_a, cue_a)
    torch.manual_seed(42)
    offline_b = extractor.extract(mix_b, cue_b)
    torch.manual_seed(42)
    a = CausalStream(extractor, cue_a)
    torch.manual_seed(42)
    b = CausalStream(extractor, cue_b)

    chunks_a = [a.push(mix_a[:71])]
    chunks_b = [b.push(mix_b[:897])]
    assert chunks_a[0].size == 0
    with pytest.raises(ValueError, match="finite mono"):
        a.push(np.array([np.nan], dtype=np.float32))
    chunks_a.append(a.push(mix_a[71:949]))
    chunks_b.append(b.push(mix_b[897:]))
    chunks_a.append(a.push(mix_a[949:]))
    chunks_b.append(b.finish())
    chunks_a.append(a.finish())
    np.testing.assert_allclose(np.concatenate(chunks_a), offline_a, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(np.concatenate(chunks_b), offline_b, atol=2e-6, rtol=1e-5)
    with pytest.raises(RuntimeError, match="ended"):
        a.push(mix_a)
    with pytest.raises(RuntimeError, match="ended"):
        b.finish()
