from speech import Wav2Vec2Recognizer

def test_import_and_config():
    r = Wav2Vec2Recognizer(noise_profile="clean")
    cfg = r.get_config()
    assert cfg["ENGINE"] == "wav2vec2"
    assert cfg["SAMPLE_RATE"] == 16000
    assert isinstance(r.SAMPLE_RATE, int)
    assert isinstance(r.MIN_SPEECH_S, float)

