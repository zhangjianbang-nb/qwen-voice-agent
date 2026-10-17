# qwen-voice-agent — 设计文档

> 2026-10-17。24/7 全双工多模态语音交互智能体。

## 1. 目标（用户规格，逐字）

| 轴 | 要求 |
|----|------|
| **感知** | 声纹采集（voiceprint）+ 人声感知（VAD / human-speech） |
| **推理** | Qwen ASR + Qwen TTS + Qwen3.8 small LLM |
| **记忆** | 声音记忆 / 工作记忆 / 语义记忆 / 情景记忆 / 程序记忆（五类） |
| **功能** | 24 小时实时监听 + 对话 + 全双工（full-duplex，可打断） |

## 2. 环境约束（实测，决定架构）

| 约束 | 实测值 | 设计对策 |
|------|--------|----------|
| 麦克风 | `arecord -l` 设备列表为空 → **本机无录音设备** | 音频源抽象为 `AudioSource`；提供 file / synthetic / null 三种，mic 为可选 extra |
| GPU torch | 已装 `torch 2.14.1+cpu`，`cuda.is_available()=False` | 核心不 import torch；本地推理走 HTTP backend，torch 只在 `transformers` extra 里惰性加载 |
| ASR/TTS/声纹模型 | 盘上无专用模型 | 每级推理藏于 backend protocol 后：`http` / `transformers` / `scripted` / `mock` |
| 音频 IO 库 | `sounddevice/soundfile/librosa` 均未装 | 只用 numpy 读写 WAV（自解析 RIFF 头）；重采样 numpy 线性 + scipy polyphase 可选 |
| 验收素材 | omni-bot 有 `silero_vad.onnx` + 两个真实语音 WAV | VAD 用 Silero ONNX（onnxruntime 已装）；验收用真实 WAV，不用合成音 |

**核心原则**：runtime core 只依赖 `numpy`。一个"没有麦克风、没有 GPU 就跑不起来"的语音 agent 是不可测试的；重依赖全部降级为 extras，由需要它的 backend 惰性 import。

## 3. 模块架构

```
qwen_voice_agent/
├── config.py          # ✅ 已完成：每级一个 dataclass，VoiceAgentConfig 组合
├── audio.py           # 帧化 / 环形缓冲 / WAV 读写 / 重采样
├── vad.py             # 能量 & Silero 后端 + 迟滞分段器（utterance segmentation）
├── voiceprint.py      # mel/onnx 嵌入后端 + 声纹库（注册/识别/排名）
├── asr.py             # ASR backend protocol: http/transformers/scripted/mock
├── tts.py             # TTS backend protocol: http/command/mock/null
├── llm.py             # LLM backend protocol: http/local/mock
├── memory/
│   ├── voice.py       # 声音记忆：谁在何时说了什么（含嵌入）
│   ├── working.py     # 工作记忆：TTL 环形，当前会话近因
│   ├── semantic.py    # 语义记忆：sqlite 事实键值
│   ├── episodic.py    # 情景记忆：jsonl 事件流
│   ├── procedural.py  # 程序记忆：json 技能/流程
│   └── manager.py     # 决定什么进 prompt（字符预算内）
├── sources.py         # AudioSource / AudioSink（file/mic/synthetic/null）
├── pipeline.py        # 全双工 turn 状态机 + barge-in + 生成 worker
├── session.py         # 24h 监督循环：idle/worker 超时、维护、优雅退出
└── cli.py             # run / doctor / enroll / transcribe / config
```

依赖方向（单向，无环）：

```
cli → session → pipeline → {sources, vad, voiceprint, asr, tts, llm, memory.manager}
                                                        memory.manager → memory.{voice,working,semantic,episodic,procedural}
```

## 4. 关键接口

### 4.1 感知

```python
# audio.py
@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray   # float32, mono, [-1, 1]
    sample_rate: int

class RingBuffer:            # 有界：ring_seconds 是 RAM 上界，不是"信任 GC"
    def push(self, samples) -> None
    def snapshot(self) -> np.ndarray

def resample(x, sr_in, sr_out) -> np.ndarray   # 优先 scipy polyphase，退化线性
def read_wav(path) -> (np.ndarray, int)        # 纯 numpy RIFF 解析，支持 int16/pcm
def write_wav(path, samples, sr) -> None

# vad.py
class VadBackend(Protocol):
    def speech_prob(self, frame: AudioFrame) -> float: ...
class EnergyVad:  ...   # 相对自适应噪声底的能量比（不是绝对电平）
class SileroVad:  ...   # onnxruntime；h/c 为 LSTM 状态（2×1×64），非单 state 张量

class Segmenter:         # 迟滞：start_frames 开、end_frames 关（end 更大）
    def push(self, frame) -> Optional[Utterance]  # 返回闭合的整段话
```

**Silero v5 ONNX contract（已探明）**：输入 `input[batch,seq] f32, sr[] i64, h[2,batch,64] f32, c[2,batch,64] f32`；输出 `output[batch,1], hn, cn`。**LSTM 状态是 h/c 两个张量**，喂 `{"input","state","sr"}` 会失败。

### 4.2 推理（三级后端，同一协议）

```python
class AsrBackend(Protocol):
    def transcribe(self, audio: AudioFrame) -> str: ...
# HttpAsr(OpenAI /v1/audio/transcriptions) | TransformersAsr | ScriptedAsr | MockAsr

class TtsBackend(Protocol):
    def synthesize(self, text: str) -> AudioFrame: ...
# HttpTts(/v1/audio/speech) | CommandTts(外部 CLI 写 WAV) | MockTts | NullTts

class LlmBackend(Protocol):
    def chat(self, messages: list[dict]) -> str: ...
# HttpLlm(/v1/chat/completions) | LocalLlm(transformers) | MockLlm
```

`system_prompt` 是带 `{context}` `{person}` 占位的模板，每轮由 memory manager 渲染 —— 人格是配置，不是改源码。

### 4.3 五类记忆

| 存储 | 载体 | 语义 | 关键约束 |
|------|------|------|----------|
| 声音 | `voice_utterances.jsonl` | 谁、何时、说了什么、嵌入 | 声纹识别结果落此，供跨会话认人 |
| 工作 | 内存 TTL 环形 | 当前会话近因（最近 N 条，TTL 内） | `working_max_items` + `working_ttl_s` 双上界 |
| 语义 | `semantic.sqlite3` | 稳定事实键值（"用户住北京"） | 可更新、可查询；`extract_facts` 从对话抽事实 |
| 情景 | `episodic.jsonl` | 事件流（某时刻发生了什么） | append-only，带时间戳 |
| 程序 | `procedural.json` | 技能/流程（"用户偏好先说结论"） | 可 seed 默认技能 |

```python
class MemoryManager:
    def context_block(self, person: str, query: str) -> str   # 字符预算内拼上下文
    def observe(self, turn: Turn) -> None                     # 一轮结束后写入各 store
    def remember_fact(self, key, value) / recall_fact(key)
```

`context_budget_chars` 是**字符**预算不是 token —— 记忆层不许依赖 tokenizer；保守取值，因为上下文块每轮前插，无界会让延迟随对话长度增长。

### 4.4 runtime

```python
# sources.py
class AudioSource(Protocol):
    def frames(self) -> Iterator[AudioFrame]: ...   # 阻塞式帧流
# FileSource(wav) | MicSource(sounddevice, extra) | SyntheticSource | NullSource
class AudioSink(Protocol):
    def play(self, audio: AudioFrame) -> None: ...
# FileSink | SpeakerSink(extra) | NullSink

# pipeline.py —— 全双工状态机
states: IDLE → LISTENING → THINKING → SPEAKING → LISTENING ...
barge-in: SPEAKING 期间若 VAD 判定用户开口 → 立即 cancel 生成 worker + 停止 TTS → 回到 LISTENING
worker: LLM+TTS 在独立线程生成；主循环持续读帧做 VAD，实现"边说边听"
class Pipeline:
    def run_turn_loop(self) -> Iterator[Turn]   # 每个完成的 Turn 产出

# session.py —— 24h 监督
class VoiceSession:
    def run(self) -> None
    # - max_seconds=None 表示永久；idle_timeout_s 无输入超时；
    # - worker_timeout_s 卡死 worker 的取消点（长跑 agent 最常见的静默死法）；
    # - maintenance_interval_s 周期性维护（记忆压缩/落盘）；
    # - SIGTERM/SIGINT → 优雅退出（保存记忆、停 TTS、关 source）
```

## 5. 验收标准 A1–A8（可编程断言）

| # | 断言 | 判据 |
|---|------|------|
| **A1** | 核心零重依赖 | 新 venv 仅装 numpy 后 `import qwen_voice_agent` 及其全部核心模块成功；`sys.modules` 不含 torch/scipy/onnxruntime/sounddevice |
| **A2** | 配置系统 | `to_json/from_json_file` 往返相等；`QVA_*` env 覆盖生效（含 bool/int/float/list 类型强转）；未知键**抛异常**而非静默丢弃 |
| **A3** | VAD 分段 | 用真实语音 WAV（`ref_minicpm_signature.wav`）跑 `Segmenter`，切出 ≥1 段、段长在 `min_speech_ms..max_speech_ms`、静音段被排除；能量后端与 Silero 后端都能给出 >0.5 的语音概率 |
| **A4** | 声纹识别 | 同一段音频注册后自识别 cosine≈1.0；不同音频（另一 WAV 的片段）识别分低于阈值；`identify` 返回完整排名 |
| **A5** | 推理后端协议 | mock/scripted/http 三种 ASR 后端各自 `transcribe` 返回 str；mock/command TTS 返回 AudioFrame；mock LLM `chat` 返回 str；http 后端在无服务时给出清晰异常而非挂死 |
| **A6** | 五类记忆 | 五个 store 各自读写 round-trip；`MemoryManager.observe` 后 `context_block` 含该轮信息且长度 ≤ `context_budget_chars`；语义事实可存可查 |
| **A7** | 全双工 + barge-in | 用合成脚本音频驱动 pipeline：TTS 播放中注入"用户说话"帧 → 生成 worker 被 cancel、TTS 停止、状态回 LISTENING；产出 Turn 序列正确 |
| **A8** | 24h 循环 | `VoiceSession` 在 `max_seconds`/`max_turns` 到期后优雅退出；worker 卡死触发 `worker_timeout_s` 取消；SIGTERM 触发优雅停机并落盘记忆；维护循环按 `maintenance_interval_s` 触发 |

每条断言写成 `tests/` 下的 pytest 用例，`pytest -q` 全绿即验收通过。

## 6. 风险与开放问题

1. **无麦克风** → 真实实时监听无法在本机跑。验收用 file source + 真实 WAV 覆盖 A3/A4/A7；mic 路径只做代码级审查 + doctor 探活。
2. **无 GPU torch** → 本地 transformers ASR/TTS 不可用；`http` backend 是生产路径，测试用 mock/scripted。
3. **无 ASR/TTS 模型** → 需定位或下载 Qwen ASR / Qwen TTS（候选：`Qwen3-ASR`、`CosyVoice2`/`Qwen3-TTS`）与声纹模型（`3D-Speaker`/`WeSpeaker`/ECAPA）。**本机可先不下载**，架构用 mock 打通；下载为后续增强项。
4. **阈值未标定** → `voiceprint.threshold=0.75` 是起点不是标定值，必须用真实注册音频按后端调。

## 7. 外部评审

按用户铁律，架构设计类任务须与 ChatGPT 对齐。评审结论回写本文件（标注"外部评审 YYYY-MM-DD"），全文归档 `docs/chatgpt-review-full.txt`。
