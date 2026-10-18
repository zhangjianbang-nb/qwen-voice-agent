# 部署记录 —— qwen-voice-agent @ 本机

> 2026-10-18。仓库 `zhangjianbang-nb/qwen-voice-agent`，部署于 `/home/whu/workspace/qwen-voice-agent`（上游 HEAD `54eafb1`）。

这份文档记录**实际怎么跑起来的**、每个后端接的是什么、实测数字是多少，以及哪些地方**没有**验证成功。

---

## 1. 一句话现状

四个推理环节全部换成真实模型并跑通端到端：**Silero VAD → FunASR SenseVoice ASR（本地）→ CAM++ 声纹（本地）→ 8082 的 Qwen3.5-4B → piper TTS（本地）→ aplay 播放**。验收测试 89 通过 / 0 失败（2 项硬件测试默认跳过，显式开启后也通过）。**未跑通的一环是本机麦克风**（见 §7）。

```bash
cd /home/whu/workspace/qwen-voice-agent
bash scripts/smoke_test.sh          # 一条命令重验整个部署
```

## 2. 环境事实（决定了架构选择）

| 项 | 实测 | 影响 |
|----|------|------|
| Python 环境 | `/home/whu/miniconda3/envs/dl`（统一环境，不新建 venv） | 包以 `pip install -e .` 装进该环境 |
| PyPI 直连 | `pypi.org/simple` 不通；**腾讯云镜像 `mirrors.cloud.tencent.com/pypi/simple` 可用** | 用它补齐依赖，见 §3 |
| GPU | `nvidia-smi` 失败（驱动只匹配内核 7.0.0-34，当前跑 7.0.0-38） | 全部走 CPU |
| CPU / 内存 | 24 核 / 93 GB；已有 qwen25vl-lite 占 ~556% CPU | 够用，但 LLM 会受争抢 |
| 采集设备 | `card 1 sofhdadsp`：`HDA Analog`(dev 0)、`DMIC Raw`(dev 6) | 见 §7 |
| 播放设备 | PipeWire 在跑；`default` 可用，直连 `plughw:1,0` 报「设备忙」 | 配置里 `output_device="default"` |
| 已有 LLM 服务 | `127.0.0.1:8082` llama-server，Qwen3.5-4B Q4_K_M，8 线程，`--reasoning-budget 0` | 直接复用为 LLM 后端 |

## 3. 安装了什么

```bash
# 包本体（离线安装：setuptools/wheel 已在环境内）
/home/whu/miniconda3/envs/dl/bin/python -m pip install -e . --no-build-isolation --no-index

# 走腾讯云镜像补齐的依赖
pip install -i https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com \
    onnxruntime        # Silero VAD / ONNX 声纹后端
    piper-tts          # 本地 TTS（piper 1.8.0）
    addict simplejson sortedcontainers kantts   # modelscope 音频 pipeline 的缺失依赖
```

`onnxruntime 1.31.0` 装好后，Silero VAD 与通用 ONNX 声纹后端才可用。

## 4. 模型（都在本机，无需下载）

| 用途 | 模型 | 位置 | 体积 |
|------|------|------|------|
| ASR | SenseVoiceSmall（FunASR） | `~/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master` | 897 MB |
| 声纹 | CAM++ `speech_campplus_sv_zh-cn_16k-common` | `~/.cache/modelscope/models/iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master` | 28 MB |
| VAD | Silero VAD v5 ONNX | `/home/whu/models/silero-vad/silero_vad.onnx` | 2.3 MB |
| TTS | piper 音色 `zh_CN-huayan-medium` | `/home/whu/models/piper/zh_CN-huayan-medium.onnx` | 63 MB |
| LLM | Qwen3.5-4B Q4_K_M（GGUF） | `/home/whu/models/Qwen3.5-4B-GGUF/`，由 8082 的 llama-server 提供 | 2.7 GB |

Silero 模型取自 `git clone https://github.com/snakers4/silero-vad`（`raw.githubusercontent.com` 直连不通，git 协议可用）。

## 5. 后端接线（`config.json`）

| 环节 | 后端 | 说明 |
|------|------|------|
| VAD | `silero` | 真实模型；能量后端保留为兜底 |
| ASR | `funasr` | 本地 SenseVoice，**不需要服务器/网络** |
| 声纹 | `funasr` | 本地 CAM++，192 维；阈值 **0.35** |
| LLM | `http` → `http://127.0.0.1:8082/v1` | 复用已有 llama-server，模型名 `Qwen3.5-4B` |
| TTS | `piper` | 本地 VITS ONNX，音色 `zh_CN-huayan-medium`，**完全离线**；`edge` 保留为音质更好的联网备选 |
| 采集 | `arecord` | 设备 `plughw:1,6`（**内置 DMIC**；`plughw:1,0` 是耳机口麦，没插耳机时静音） |
| 播放 | `aplay` | 设备 `default`（PipeWire），可被打断 |

`echo_guard=true`：单机扬声器+麦克风时，agent 会听见自己 → 自打断。开启后播放期间丢弃采集帧；代价是播放时无法真打断。戴耳机可设 `false` 换回真全双工（见 §7）。

**声纹门（主人 vs 电视）**：`wake.require_enrolled=true` 让唤醒/响应必须过 CAM++；`vad.barge_in_verify=true` 让打断也必须过声纹——电视里的人声不会唤醒助手，也不会在生成期间把回复掐断。两条路径都 fail closed（声纹出错 = 不是主人）。注册主人声纹：

```bash
bash scripts/enroll_owner.sh     # 录 15 s，确认转写后注册，并重启 agent
```

注册前 `state/voiceprints.json` 只有示例音色 Ada/Bo，agent 对任何声音都不响应（安全默认值）。

## 6. 实测数字

**单轮延迟（暖机后）**

| 环节 | 延迟 | 备注 |
|------|------|------|
| ASR（SenseVoice） | **83 ms** | 3.7 s 音频，约 37× 实时 |
| 声纹（CAM++） | **19 ms** | 192 维嵌入 |
| LLM（Qwen3.5-4B, CPU） | 823–2603 ms | 受 CPU 争抢影响明显 |
| TTS（piper） | **38–80 ms** | 本地 VITS，无网络 |
| 合计 | **约 2.6–2.8 s** | 模型在启动时 warmup，首轮无额外加载 |

**声纹判别力**（CAM++ 自带示例音频）

| 对比 | cosine |
|------|--------|
| 同一人不同话句 | **0.6936** |
| 另一人 | 0.0072 |
| 未注册的合成音（陌生人） | **0.0206 → 拒绝** |

mel 占位后端做不到拒识陌生人，CAM++ 可以——这是本次替换的核心收益。

**播放打断**：6.00 s 音频，0.5 s 后 cancel → `play()` 0.50 s 返回（延迟 ≤1 个 100 ms 分块）；无 cancel 时 1.00 s 音频耗时 1.36 s。

**真实播放链路已核实**：`--sink aplay` 运行期间，两个 `aplay` 子进程各从 stdin 读走 166 KB / 131 KB PCM（≈5.2 s / 4.1 s），`/proc/<pid>/io` 的 `rchar` 随时间单调增长（实时消费而非一次灌入）；其 fd 是两条 PipeWire socket 而非 `/dev/snd/*`，因为 `-D default` 走 PipeWire 插件；`wpctl status` 显示默认 sink 为 `sof-hda-dsp Speaker + Headphones [vol: 1.26]`，未静音。

**测试**：`pytest` 89 passed / 2 skipped；`QVA_TEST_AUDIO=1` 时那 2 项也通过。

**端到端 demo**（`--source file --sink aplay --no-loop`，句间隔 12 s）：两轮均 `interrupted=False`，第 1 轮正确引用第 0 轮的回复（working memory 生效），两次完整播放，exit=0。注意句间隔是功能参数而非审美——见 `docs/DESIGN.md` §9.5：本机一轮（ASR→LLM→TTS→实时播放）约 9.7 s，间隔 7 s 正好压在 barge-in 边界上，同一份音频不同次运行会时而打断时而完整。

## 7. 已知限制（没跑通的部分，如实记录）

1. ~~**本机麦克风拿不到信号**~~ —— **2026-10-18 当天已推翻，麦克风是好的**。修正记录如下，避免后人重蹈：
   - 内置麦 = **DMIC Raw `plughw:1,6`**。连续采 10 秒，每秒 `rms` 170–334、峰值 925–1878；房间有音频时 `rms` 450–1300、峰值 3 万+。`transcribe` 能正确转写出拾到的语音。`--source mic` + `input_device="plughw:1,6"` 已实跑出多轮对话。
   - `plughw:1,0` 是**耳机口麦**（HDA Analog），没插耳机 → `rms≈0.00004`。"静音"的是这条路径，不是内置麦。
   - **原判断错在哪**：我用 `arecord` 的 `/proc/<pid>/io` 里 `rchar` 不增长来判"没有数据"。但 `plughw` 走 ALSA mmap，`arecord` 用 memcpy 从映射缓冲区取数，**不产生 `read()` 系统调用，`rchar` 本来就不涨**。判断采集在不在流要看 `/proc/asound/card1/pcm6c/sub0/status`：`state: RUNNING` + `hw_ptr` 前进 + `avail > 0`。
   - **真正的阻塞是独占**：DMIC 同一时刻只能被一个进程打开（`dsnoop:1,6` 也开不了已被独占的 slave，报"设备或资源忙"）。当天它被 `qwen25vl-agent web` 的 `arecord` 长期独占，而那个父进程有守护逻辑——杀掉子进程约 20 秒后会重生一个继续占。已按用户指示停掉 `qwen25vl-agent`(8090) 与 `qwen25vl-lite`(8081)，麦克风归 qwen-voice-agent 独占。

2. **环境噪声会盖过用户** —— 本机 `RustDesk` 有两条活跃播放流推到 `Speaker + Headphones`，麦克风会把它的远程音频整段拾进来。实测在噪声下 87 轮里 80 轮被 `barge-in` 打断、20 轮回复为空——不是 bug，是"连续语音"下的正确行为。要它只听你说话：把 RustDesk 音频静音（`wpctl set-mute`）或戴耳机（并把 `vad.echo_guard` 设 `false` 恢复真全双工）。

3. ~~**TTS 依赖网络**~~ —— 已用 `piper` 本地合成解决（38–80 ms/句，断网可用）。`edge` 后端仍保留：音质更自然，但合成时需连 Microsoft 服务器，网络抖动会让 TTS 从 1.4 s 涨到 2.8 s。modelscope SambertHifigan 因依赖链太长（缺 `kantts` 之外的其他组件）未采用。

4. **单机自激（回声）** —— 扬声器+麦克风同机时 agent 会听见自己。已实现 `echo_guard` 缓解，但这是半双工妥协，不是真 AEC。真正的修法是播放参考信号 + AEC，仓库 roadmap 的 P1 项仍未做。

5. **CPU 争抢** —— 2026-10-18 当天实测：同时跑着 `qwen25vl-lite`（27.8 GB RSS / 实时 80% CPU，还挂着 ~180 个 CLOSE-WAIT 泄漏连接）时，一轮的 LLM 从 0.8–2.3 s 涨到 **7.2 s**，端到端"哑音"达 8.7 s。按用户指示停掉 `qwen25vl-agent`(8090) 与 `qwen25vl-lite`(8081) 后，内存 40→11 GB、load 4.5→1.30，LLM 回到亚秒级。要稳定低延迟就别和它们同跑，或换 GPU（当前 GPU 不可用）。

6. **`~/.local/lib/python3.12/site-packages` 会盖住 conda 环境** —— numpy 从该路径解析（版本同为 2.5.3，暂未出问题），但装包时要意识到这个优先级。

## 8. 常用命令

```bash
cd /home/whu/workspace/qwen-voice-agent

qwen-voice-agent -c config.json doctor            # 体检：依赖/设备/模型/LLM 服务
qwen-voice-agent -c config.json transcribe a.wav   # 用 SenseVoice 转写
qwen-voice-agent -c config.json enroll a.wav --person-id p1 --name Ada
qwen-voice-agent -c config.json identify b.wav     # 含完整排名

# 有界试跑（文件源，抓取 agent 语音）；--no-loop 让文件只放一遍并等最后一句答完
qwen-voice-agent -c config.json run --source file --source-path state/smoke/dialog.wav \
    --sink file --sink-path state/smoke/agent_out.wav --no-loop

# 有声 demo（真扬声器播放）
QVA_TEST_GAP=12.0 python scripts/build_test_audio.py state/demo
qwen-voice-agent -c config.json run --source file --source-path state/demo/dialog.wav \
    --sink aplay --memory-root state/demo/mem --no-loop

# 24/7 实时（麦克风+扬声器）
bash scripts/run_agent.sh
```

运行态文件在 `state/`（五类记忆 + `voiceprints.json`）。`scripts/run_agent.sh` 的所有路径都可用环境变量覆盖（`QVA_PYTHON` / `QVA_CONFIG` / `QVA_SOURCE` / `QVA_SINK`），便于 systemd 托管。
