# Test assets

Real speech is what makes the perception acceptance tests (A3/A4) mean anything —
a synthetic tone proves the plumbing, not that the detector works on a voice. The
clips here are the ones the CAM++ checkpoint ships as its own examples, so the
voiceprint tests are calibrated against the model author's data.

| File | Source |
|------|--------|
| `ref_minicpm_signature.wav` | `speaker1_a_cn_16k.wav` from `iic/speech_campplus_sv_zh-cn_16k-common` |
| `ref_en_dlc_1.wav` | `speaker2_a_cn_16k.wav` from the same checkpoint (a different speaker) |
| `silero_vad.onnx` | `src/silero_vad/data/silero_vad.onnx` from https://github.com/snakers4/silero-vad |

Both checkpoints carry their own licences; these files are here for local test runs
only and are not redistributed by this project. Delete them and the dependent tests
skip rather than fail — `conftest.py` treats a missing asset as "not available on
this machine", not as a broken build.
