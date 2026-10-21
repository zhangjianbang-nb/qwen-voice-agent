# 部署记录 —— qwen-voice-agent @ 本机

> 2026-10-18。仓库 `zhangjianbang-nb/qwen-voice-agent`，部署于 `/home/whu/workspace/qwen-voice-agent`（上游 HEAD `54eafb1`）。

这份文档记录**实际怎么跑起来的**、每个后端接的是什么、实测数字是多少，以及哪些地方**没有**验证成功。

---

## 1. 一句话现状

四个推理环节全部换成真实模型并跑通端到端：**Silero VAD → FunASR SenseVoice ASR（本地）→ CAM++ 声纹（本地）→ 8082 的 Qwen3.5-4B（已全量 offload 到 GPU）→ piper TTS（本地）→ aplay 播放**，并接上了 **OpenAI 风格工具调用环**。工具集从 3 个扩展到 **15 个**：时间/记忆、文件读写、浏览器/应用、桌面输入（opt-in）、定时提醒/任务。验收测试 **129 通过 / 2 跳过**（2 项硬件测试默认跳过，显式开启后也通过）。

2026-10-19 的关键变化：GPU 驱动恢复（`nvidia-smi` 可用，RTX 5060 8 GB），LLM 单轮从 **2603 ms 降到 297 ms**；工具调用端到端实测 3/3 命中，单轮推理链 **0.4–1.1 s**。本机麦克风此前已验证可用（见 §7.1）。同日新增的动作工具与定时提醒均做了实机验证，结果见 §6。

```bash
cd /home/whu/workspace/qwen-voice-agent
bash scripts/smoke_test.sh          # 一条命令重验整个部署
```

## 2. 环境事实（决定了架构选择）

| 项 | 实测 | 影响 |
|----|------|------|
| Python 环境 | `/home/whu/miniconda3/envs/dl`（统一环境，不新建 venv） | 包以 `pip install -e .` 装进该环境 |
| PyPI 直连 | `pypi.org/simple` 不通；**腾讯云镜像 `mirrors.cloud.tencent.com/pypi/simple` 可用** | 用它补齐依赖，见 §3 |
| GPU | **2026-10-19 恢复可用**：`nvidia-smi` 正常，RTX 5060 Laptop 8 GB，驱动 595.99.02 / CUDA 13.2 | llama.cpp 用 CUDA 后端，`NGLAYERS=99` 全量 offload；torch 仍是 CPU 版，不影响 |
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
| LLM | `http` → `http://127.0.0.1:8082/v1` | 复用已有 llama-server，模型名 `Qwen3.5-4B`；**用 `NGLAYERS=99` 启动即全量 GPU offload**（约 3.2 GB 显存） |
| TTS | `piper` | 本地 VITS ONNX，音色 `zh_CN-huayan-medium`，**完全离线**；`edge` 保留为音质更好的联网备选 |
| 采集 | `arecord` | 设备 `plughw:1,6`（**内置 DMIC**；`plughw:1,0` 是耳机口麦，没插耳机时静音） |
| 播放 | `aplay` | 设备 `default`（PipeWire），可被打断 |
| 工具 | `tools.builtins`（15 个） | 文件工具沙箱在 `agent_files/`；浏览器只收 http/https；应用走 `allowed_apps` 白名单；桌面输入需 `control_enabled=true`；**没有 shell/exec 工具** |
| 定时提醒/任务 | `Scheduler` + JSON | `set_reminder` 播报固定文字；`schedule_task` 到点跑 LLM+工具并播报结果。时间支持“10分钟后 / 明天早上8点 / 每天8点 / 周一9点 / ISO”；pipeline 空闲时主动播报，可被插话打断 |

`echo_guard=true`：单机扬声器+麦克风时，agent 会听见自己 → 自打断。开启后播放期间丢弃采集帧；代价是播放时无法真打断。戴耳机可设 `false` 换回真全双工（见 §7）。

**声纹门（主人 vs 电视）**：`wake.require_enrolled=true` 让唤醒/响应必须过 CAM++；`vad.barge_in_verify=true` 让打断也必须过声纹——电视里的人声不会唤醒助手，也不会在生成期间把回复掐断。两条路径都 fail closed（声纹出错 = 不是主人）。注册主人声纹：

```bash
bash scripts/enroll_owner.sh     # 录 15 s，确认转写后注册，并重启 agent
```

注册前 `state/voiceprints.json` 只有示例音色 Ada/Bo，agent 对任何声音都不响应（安全默认值）。

## 6. 实测数字

**单轮延迟（暖机后，2026-10-19 GPU 复测）**

| 环节 | 延迟 | 备注 |
|------|------|------|
| ASR（SenseVoice） | **74–118 ms** | 本地 CPU |
| 声纹（CAM++） | **19–27 ms** | 192 维嵌入 |
| LLM（Qwen3.5-4B, GPU 99 层） | **217–297 ms** | 单轮；原 CPU 版 823–2603 ms |
| LLM（含工具环，两轮） | **698–955 ms** | 第一次 `tool_calls` + 工具执行 + 第二次生成 |
| TTS（piper） | **24–150 ms** | 本地 VITS，无网络 |
| 合计（无工具） | **约 0.4–0.5 s** | 从语音结束到可播放 |
| 合计（含工具） | **约 0.8–1.1 s** | 查时间/记事实/查事实 |

GPU offload 是这一轮最大的单项收益：同一个 8082 服务、同一个模型，只把 `NGLAYERS` 从 0 改成 99，LLM 就从 2.6 s 降到 0.3 s。启动命令：

```bash
THREADS=8 NGLAYERS=99 bash /home/whu/workspace/qwen25vl-agent/scripts/serve_qwen35.sh
```

**工具调用实测**（`state/tools/dialog.wav`，三句话，GPU 服务）：

| 用户 | 工具 | 最终回复 |
|------|------|----------|
| 现在几点了？ | `get_current_time({})` → `2026-10-19 10:36:48（星期一）` | 现在是2026年10月19日星期一上午10点36分。 |
| 记住我叫小明。 | `remember_fact({"key":"name","value":"小明"})` → `已记住` | 记住了。 |
| 我叫什么名字？ | `recall_fact({"key":"name"})` → `小明` | 你叫小明。 |

三轮全部 `tools=1`、`interrupted=False`，语义记忆里落下了 `name=小明`。提示词必须把工具规则放在人格描述**之前**，并明确“只说记住了而不调用工具等于没记住”——否则 4B 模型会直接编一句“好的，记住了”，不调工具（实测 5/5 都是这样）。

**动作工具与定时提醒实测（2026-10-19，真实配置）**

| 场景 | 结果 |
|------|------|
| `write_file` / `read_file` / `list_files` | 写入 `agent_files/notes/verify.txt`（59 B），读回内容一致，列目录看到文件 |
| 越界路径 | `../escape.txt`、工作区外的绝对路径、指向外部的 symlink 全部被拒（`path escapes the workspace`） |
| `open_browser` | `https://example.com` 已在现有 Chrome 中打开（`xdg-open` 返回 0） |
| `set_reminder` / `list_reminders` / `cancel_reminder` | 设置“2分钟后喝水”→ 列表显示 id 与时间 → 取消成功 |
| `schedule_task` | “每天早上8点播报天气”→ 创建 `kind=task`、`repeat=daily` 的任务；LLM 选择正确（不会误用 set_reminder） |
| `screenshot` | **失败但如实报错**：`no screenshot backend: install gnome-screenshot (GNOME/Wayland) or grim (wlroots)`；本机是 GNOME/Wayland，scrot 只能拍到黑色 X root |
| `type_text` / `press_key` | 在 Xephyr 的 X11 会话里对 Tk 输入框实测：工具输入 `hello via tool` + `Return`，程序收到并退出，输出文件内容为 `hello via tool` |
| 实时麦克风全链路（2026-10-19） | 临时取消扬声器静音并置音量 1.0，播放放大后的 `dialog_loud.wav`；测试配置 `input_gain=8.0`。原始 DMIC 信号 rms≈0.0055、peak≈0.12，Silero 在 gain=1 时 max_prob≈0.10，gain=8 时 max_prob≈0.97；agent 产出 1 轮 `快介绍一下。` → `我是你的语音助手，随时帮你处理各种任务。`，写出 156 KB WAV。默认 `input_gain=1.0` 对扬声器回放偏小，真人近距离说话需按实际电平校准。 |
| 主动提醒全链路 | 预置“喝水”到临时 `schedules.json`，`--source file --sink file --no-loop --memory-root /tmp/qva-reminder-verify`：日志 `schedule due: ... '喝水'`，CLI 输出 `[0] [提醒 ...] -> '提醒你：喝水'`，生成 71 KB WAV，一次性提醒随后从 JSON 删除 |
| 定时任务全链路 | 预置 `kind=task` 的“现在几点了？”到期任务：日志 `schedule due: ... (task)`，两次 LLM 请求（工具环 `get_current_time` + 最终回答），CLI 输出 `[0] [任务 ...] -> '现在是2026年10月19日星期一中午12点49分。'`，生成 235 KB WAV，任务随后删除 |

**4B 模型工具选择实测**（8082 GPU 服务，15 个工具 schema 全量下发）：**16/16 命中**（含取消提醒），覆盖查时间、记忆读写、文件读写/列目录、打开浏览器（URL 与搜索两种）、打开应用、输入文字、按键、截图、设提醒、定时任务、列提醒、取消提醒。

最早验证的 6 个场景分别调用了 `get_current_time`、`write_file`、`set_reminder`（到点提醒）、`schedule_task`（每天播报天气）、`open_browser`（query→百度搜索 URL）、`list_reminders`。`set_reminder` 的结果格式改为“今天 12:16”后，模型不再把当天 12:16 说成“明天”。

**声纹判别力**（CAM++ 自带示例音频）

| 对比 | cosine |
|------|--------|
| 同一人不同话句 | **0.6936** |
| 另一人 | 0.0072 |
| 未注册的合成音（陌生人） | **0.0206 → 拒绝** |

mel 占位后端做不到拒识陌生人，CAM++ 可以——这是本次替换的核心收益。

**播放打断**：6.00 s 音频，0.5 s 后 cancel → `play()` 0.50 s 返回（延迟 ≤1 个 100 ms 分块）；无 cancel 时 1.00 s 音频耗时 1.36 s。

**真实播放链路已核实**：`--sink aplay` 运行期间，两个 `aplay` 子进程各从 stdin 读走 166 KB / 131 KB PCM（≈5.2 s / 4.1 s），`/proc/<pid>/io` 的 `rchar` 随时间单调增长（实时消费而非一次灌入）；其 fd 是两条 PipeWire socket 而非 `/dev/snd/*`，因为 `-D default` 走 PipeWire 插件；`wpctl status` 显示默认 sink 为 `sof-hda-dsp Speaker + Headphones [vol: 1.26]`，未静音。

**测试**：`pytest` **129 passed / 2 skipped**（`tests/test_tools.py` 27 项，`tests/test_scheduler.py` 13 项）；`QVA_TEST_AUDIO=1` 时那 2 项也通过。

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

5. **CPU 争抢** —— 2026-10-18 实测：同时跑着 `qwen25vl-lite`（27.8 GB RSS / 实时 80% CPU）时，一轮的 LLM 从 0.8–2.3 s 涨到 **7.2 s**。停掉后回到亚秒级。**2026-10-19 GPU 恢复后这个问题基本消失**（LLM 在 GPU 上 0.2–0.3 s），但 ASR/声纹/piper 仍在 CPU，重负载仍会拖慢它们。

6. **工具调用的两个边界** —— (a) 模型可能“假装记住”：不调 `remember_fact` 却说“记住了”，结果是下一轮查不到。已用“规则前置 + 负例”的提示词解决，但换模型/换语言后要重新验证；(b) 工具执行期间 `_task_busy=true`，环境人声不会开启新 turn（除非外部 `task_gate` 放行），这是有意为之——任务执行中不该被电视打断。

7. **`~/.local/lib/python3.12/site-packages` 会盖住 conda 环境** —— numpy 从该路径解析（版本同为 2.5.3，暂未出问题），但装包时要意识到这个优先级。

8. **Wayland 桌面控制只覆盖 X11/XWayland** —— 本机是 GNOME/Wayland，Chrome 用 `--ozone-platform=wayland`。`xdotool` 能定位窗口并对 XWayland 客户端注入按键（在 Xephyr 里已实测成功），但原生 Wayland 客户端会忽略合成事件；`windowactivate --sync` 在部分窗口管理器下会挂起，所以工具改为“有界 `windowactivate` + `windowfocus` + XTEST”，失败才退回 `--window`。要在原生 Wayland 上真正注入输入，需要 `ydotool`（依赖 `/dev/uinput` 权限）或 RemoteDesktop portal，本机均未配置。

9. **截图后端缺失** —— `scrot` 在 Wayland 下只能拍到黑色 X root（实测 `/tmp/_scrot_test.png` 全黑）；`gnome-screenshot`/`grim` 未安装，GNOME Shell 的 Screenshot D-Bus 对非 shell 调用返回 `AccessDenied`。工具会明确报错而不是返回一张黑图。装 `gnome-screenshot` 或 `grim` 后即可用。

10. **主动提醒的边界** —— (a) 只在 pipeline 空闲（无生成、无未闭合语句、上一帧 VAD 为静音）时播报，用户正在说话时会顺延；(b) 被用户插话打断后按 `scheduler.retry_s`（默认 60 s）重试，不会立刻重播；(c) 重复任务按本地墙钟推进（`daily`/`weekly`/`weekdays`），进程停机跨过多次触发时只补播一次，然后推进到下一个未来时间；(d) `control_enabled=true` 时 `type_text` 会向当前焦点窗口输入，误触可能打到别的窗口——不需要桌面控制时应关掉。
11. **定时任务的执行代价** —— `kind=task` 在到点后才跑 LLM + 工具环，因此延迟比固定提醒高（本机实测两次 LLM 请求约 0.9 s，再加 piper 合成）；LLM/工具失败时按 `scheduler.retry_s` 重试，不会静默丢失。任务 prompt 同样受文件沙箱、http/https、应用白名单约束，但它可以调用动作工具，所以不要把 `schedule_task` 暴露给不可信说话人。

## 8. 常用命令

```bash
cd /home/whu/workspace/qwen-voice-agent

qwen-voice-agent -c config.json doctor            # 体检：依赖/设备/模型/LLM 服务
qwen-voice-agent -c config.json tools             # 列出 15 个工具名与 JSON schema
qwen-voice-agent -c config.json transcribe a.wav   # 用 SenseVoice 转写
qwen-voice-agent -c config.json enroll a.wav --person-id p1 --name Ada
qwen-voice-agent -c config.json identify b.wav     # 含完整排名

# 定时提醒有界验证：先往临时 memory root 的 schedules.json 放一条到期提醒
python - <<'PY'
import time
from qwen_voice_agent.config import SchedulerConfig
from qwen_voice_agent.scheduler import Scheduler
Scheduler(SchedulerConfig(store_path="/tmp/qva-reminder/schedules.json")).add("喝水", time.time()-1)
PY
qwen-voice-agent -c config.json run --source file --source-path /tmp/qva-reminder/silence.wav \
    --sink file --sink-path /tmp/qva-reminder/out.wav --memory-root /tmp/qva-reminder \
    --no-loop --no-wake --anyone --seconds 12
# 日志应出现 "reminder due"，CLI 应打印 "[提醒 <id>] -> '提醒你：喝水'"，out.wav 非空

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

运行态文件在 `state/`（五类记忆 + `voiceprints.json` + `schedules.json`）。`scripts/run_agent.sh` 的所有路径都可用环境变量覆盖（`QVA_PYTHON` / `QVA_CONFIG` / `QVA_SOURCE` / `QVA_SINK`），便于 systemd 托管。
