# VMR Wiki

将 QVHighlights 与 UCA-VMR 视频转换成固定的视觉时间线，再让每条 Query 在全新的 Codex / Claude Code 进程中完成 Video Moment Retrieval。实现范围对应 [PLAN.md](PLAN.md) 的六个 MVP milestones。

```text
QVHighlights / UCA-VMR annotations → videos / queries / ground_truth manifests
视频 → 固定采样 → VLM captions → Wiki → SHA256 Freeze
单 Query + 当前 Wiki → 独立容器 / 新进程 → prediction.json
predictions → aggregate → mAP / R@K → metrics.json
```

Ground Truth 和预测始终使用 `moments: [...]`，不会在转换或聚合时丢弃额外片段。

## 安装

需要 Python 3.10+、宿主机的 `ffmpeg` / `ffprobe`，以及运行中的 Docker。命令均在本仓库根目录执行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

macOS 可通过 `brew install ffmpeg` 安装媒体工具；Linux 可使用发行版包管理器。运行时只需要 PyYAML，VLM 客户端使用 Python 标准库。

构建 Agent 镜像时必须指定 CLI 版本；所有实验会记录实际镜像的 SHA256 ID，在一次实验中始终按此 ID 启动，避免浮动 tag 造成混用。

```bash
docker build -f docker/Dockerfile \
  --build-arg CODEX_VERSION=0.153.4 \
  --build-arg CLAUDE_VERSION=2.1.158 \
  -t vmr-wiki-agents:local .
```

镜像中只安装 CLI 和基本文件工具，不包含仓库、数据集或认证文件。CLI 版本可以调整，但必须支持相应 adapter 使用的参数，并为变更后的运行创建新实验。

## 配置

编辑 [config.yaml](config.yaml)：

- `ingest.vlm.model`：明确指定支持图像输入、`temperature` 和 `max_tokens` 的 Chat Completions 模型。
- `query.model`：明确指定所使用的 Agent 模型。
- `ingest.vlm.base_url`：默认 OpenAI `/v1`，也支持兼容的 VLM API。
- `ingest.vlm.api_key_env`、`query.api_key_env`：只填写环境变量名，不填写密钥。
- `sample_interval_sec`、预处理尺寸、prompt、temperature、token 上限和 `max_predictions` 是固定实验变量。

通过环境配置 `OPENAI_API_KEY`（Ingest）、`CODEX_API_KEY`（Codex）或 `ANTHROPIC_API_KEY`（Claude Code）。可以在配置中指定其他变量名。当前 Query adapter 使用 API key，不挂载宿主机登录状态。

配置中的相对目录均相对于配置文件所在目录解析。保存到实验中的配置使用完整路径，不包含环境变量的值。模型名保留显式占位符，未配置时不会发起请求。

## 1. 转换数据集

准备 QVHighlights 官方 **train 或 val** JSONL annotation，以及对应的 `.mp4` 文件。输入字段为 `qid`、`vid`、`query`、`duration`、`relevant_windows`。视频文件必须名为 `<vid>.mp4`，且已经裁成 annotation 中对应的片段；文件内 0 秒就是标注的 0 秒，不能直接提供未裁剪的长视频。

```bash
python adapters/qvhighlights.py \
  --annotations /path/to/highlight_val_release.jsonl \
  --video-root /path/to/qvhighlights/videos \
  --output datasets/qvhighlights
```

生成：

```text
datasets/qvhighlights/
├── videos.jsonl
├── queries.jsonl
├── ground_truth.jsonl
└── dataset_metadata.json
```

所有 GT windows 原样保留；`queries.jsonl` 只含 `query_id`、`video_id`、`query`。重复 ID、非法时间区间、同视频时长不一致、缺少标签都会报错，已有输出目录不会被覆盖。官方 test split 的 GT 不公开，因此当前转换入口仅接受带标签的 train/val。

转换器写入绝对视频路径；手工编写 manifest 时，相对 `video_path` 以 `videos.jsonl` 所在目录为基准。

### 1.1 UCA-VMR

UCA-VMR 的 annotation 是每条 Query 一行 JSONL，输入字段为 `query_id`、`video_id`、`path`、`duration`、`query`（对象，取 `text` 字段）、`gold_moments`（`[[start, end], ...]`）。`path` 是相对于 `--video-root` 的视频路径，例如 `Videos/Abuse/Abuse043_x264.mp4`。每个 Query 只有一个 gold moment，原样保留在 `moments` 数组中。

```bash
python adapters/uca.py \
  --annotations /path/to/UCA-VMR/dev.jsonl \
  --split dev \
  --video-root /path/to/UCF_Crimes \
  --output datasets/uca
```

用 `--annotation-dir` 可一次发现目录下全部 `<split>.jsonl`：

```bash
python adapters/uca.py \
  --annotation-dir /path/to/UCA-VMR \
  --video-root /path/to/UCF_Crimes \
  --output datasets/uca \
  --default-eval-split dev
```

生成：

```text
datasets/uca/
├── videos.jsonl
├── queries.jsonl
├── ground_truth.jsonl
└── dataset.json
```

`queries.jsonl` 只含 `query_id`、`video_id`、`query`、`split`；`ground_truth.jsonl` 保留全部 `gold_moments`。重复 ID、非法时间区间、同视频时长不一致、`path` 逃出 video root、`query.text` 缺失都会报错，已有输出目录不会被覆盖。UCA-VMR 的 test split GT 公开，因此 train / dev / test 均可转换。

## 2. Ingest 与 Freeze

单视频入口：

```bash
python harness/ingest.py \
  --video /path/to/video.mp4 \
  --video-id video_001 \
  --output wiki/qvhighlights/video_001
```

正式数据集建议批量执行并冻结：

```bash
python harness/ingest_all.py --dataset qvhighlights --freeze
```

也可以在检查 Wiki 后单独冻结：

```bash
python harness/freeze.py --dataset qvhighlights
```

Ingest 只读取 `videos.jsonl`，按 `0, interval, 2 × interval, … < duration` 抽帧，每张图独立调用同一个固定 VLM prompt。时间戳表示请求的采样时刻，解码器选取该时刻对应的可解码视频帧；它不是事件的精确边界。图像保持纵横比，并限制最长边，不放大小图像。

每个视频输出：

```text
wiki/qvhighlights/<video_id>/
├── wiki.md
├── frames.jsonl
├── frames/
├── ingest.json
└── frozen.json
```

`ingest.json` 记录媒体 SHA256、配置、FFmpeg 版本和内容哈希；`frozen.json` 覆盖 Markdown、JSONL、**每一张图像**及 Ingest 元数据。数据集级 `freeze.json` 固定全部视频的版本。

完整 Ingest 再次执行时只核验并复用，不重新 caption。失败的临时输出被清除；API 仅对临时网络错误、限流和服务端错误做有限重试。并发 Ingest 同一个视频会被锁拒绝。Freeze 后目录只读，修改配置或视频必须使用新的 Wiki 根目录。批量 Ingest / Freeze 校验媒体与 annotation 时长，差异超过 0.1 秒时拒绝继续，不会静默缩放标签。

## 3. 单 Query / 批量 Query

```bash
python harness/run_query.py \
  --dataset qvhighlights \
  --agent codex \
  --experiment codex_wiki_v1 \
  --query-id 7803

python harness/run_all_queries.py \
  --dataset qvhighlights \
  --agent codex \
  --experiment codex_wiki_v1
```

运行 Claude Code 时用独立实验名，并设置相应模型：

```bash
python harness/run_all_queries.py \
  --dataset qvhighlights \
  --agent claude_code \
  --model YOUR_CLAUDE_MODEL \
  --experiment claude_wiki_v1
```

每条 Query 创建唯一临时 workspace，内容严格为：

```text
AGENTS.md
task.json
wiki/wiki.md
wiki/frames.jsonl
wiki/frames/
output/
```

宿主仓库、视频、GT、其他 Query、其他预测都不挂载进容器。容器以非 root 用户运行，根文件系统和整个 workspace 只读，唯一的任务输出挂载点 `output/` 可写。容器 Home 和临时目录每次新建且退出销毁，CLI 禁用 session 持久化，不使用 resume。Claude 显式加载同一份 `AGENTS.md`。

Docker 提供文件系统和进程隔离，**不是完全离线环境**：Agent CLI 需要联网访问模型 API，当前未实现按域名限制网络出口。模板明确只允许利用已提供的 Wiki，禁用联网检索；Claude 仅开放文件及 shell 内置工具，不启用额外 MCP。这里的保证是宿主实验数据不被挂载，而不是针对恶意 Agent 的网络隔离证明。

Harness 等待进程结束、验证输入未变、校验输出，再保存结果并清除 workspace。超时会强制删除整个容器和进程。JSON 缺失、解析失败、字段多余/缺失、错误 ID、布尔值冒充数字、NaN、时间越界、分数越界、排序错误、预测数量超限、额外输出文件等均记录为失败，**不会自动修复或重新调用 Agent**。`moments: []` 是成功的 abstention。

已有运行记录（成功或失败）不会再执行。再次运行同一批命令只处理尚未开始的 Query；中断后留下的 `running` 记录在再次读取时转为失败。配置、模型、Agent、代码、模板、manifest 或镜像 ID 变化时，必须使用新实验名。

```text
results/<experiment>/
├── predictions/<query_id>.json
├── run_metadata/<query_id>.json
├── logs/<query_id>.stdout.log
├── logs/<query_id>.stderr.log
├── templates/
├── experiment.json
└── config.yaml
```

元数据包含时间、exit code、失败原因、Wiki/config/template/source/manifest 哈希、镜像 ID 和 Git commit；目录不是 Git 仓库时 commit 为 `null`，代码仍有内容哈希。

若宿主进程被 `SIGKILL` 或机器断电，正常的 `finally` 清理无法执行。确认没有相关进程/容器继续运行后，可人工移除对应 `.experiment.lock`、`.ingest.lock` 和残留临时目录；不要删除已记录的 Query 结果来假装首次执行。

## 4. 聚合与评测

即使批量运行因部分 Query 失败而返回非零 exit code，也应继续聚合与评测。

```bash
python harness/aggregate.py \
  --input results/codex_wiki_v1/predictions \
  --output results/codex_wiki_v1/predictions.jsonl

python harness/evaluate.py \
  --pred results/codex_wiki_v1/predictions.jsonl \
  --gt datasets/qvhighlights/ground_truth.jsonl
```

默认写入相邻的 `metrics.json`；可通过 `--output` 指定位置。评测读取实验保存的配置和 run metadata；数据集元数据选择 QVHighlights evaluator，或显式使用 `--evaluator generic`。UCA-VMR 的 `dataset.json` 中 `evaluator` 为 `generic`，因此直接使用通用 evaluator，无需 `--evaluator` 参数。

**所有指标的单位为百分数，范围 0–100。** 分母是 GT manifest 中的全部 Query，缺失/失败结果按零命中处理；不因某个 Query 失败就从评测集删除它。只跑一个 Query 时直接用完整 GT 评测，其余未运行的 Query 也会计入失败；需要子集实验时应先准备对应子集的 manifests。

通用 evaluator 输出配置的 `R@K,IoU=T`，语义为前 K 个预测命中任意一个 GT 即成功。UCA-VMR 使用此 evaluator（每个 Query 只有一个 GT moment，属于 `any_acceptable_moment` 语义）。QVHighlights adapter 额外输出：

- `MR-full-mAP`：主指标，IoU 0.50 到 0.95、间隔 0.05 的平均 AP。
- `MR-full-mAP@0.5`、`MR-full-mAP@0.75`，以及 R1@0.5 / R1@0.7。
- short `(0,10]`、middle `(10,30]`、long `(30,150]` 秒的分组 mAP。

QVHighlights AP 使用前 10 个预测，按置信度逐一与尚未匹配的 GT 做匹配；重复命中同一个 GT 不能重复加分。实现遵循官方 moment retrieval 语义，不计算与本任务无关的 highlight/saliency 指标。没有 Query 的长度组返回 JSON `null`。

输出同时包括 `failed_runs`、失败 ID、缺失数量、abstention 数量、每条 Query 平均预测数、top K 和 IoU thresholds。聚合保留所有预测和 evidence，不会静默降成 top-1，也不会重排或修复不合法结果。

### 使用官方代码

优先使用官方 evaluator 时，提供本地 [moment_detr](https://github.com/jayleicn/moment_detr) checkout：

```bash
python -m pip install -e '.[official]'
python harness/evaluate.py \
  --pred results/codex_wiki_v1/predictions.jsonl \
  --gt datasets/qvhighlights/ground_truth.jsonl \
  --official-root /path/to/moment_detr
```

此路径在独立 evaluator 进程中调用官方 `compute_mr_ap` / `compute_mr_r1`，记录官方源码 SHA256；官方执行失败会明确报错，不静默降级。未指定官方 checkout 时使用本地 dataset adapter，指标中标注 `implementation`。

上游 R1 不能处理空预测，AP 会遗漏空列表。因此官方桥接仅在评测器内将缺失/空预测表示成零长度、零分数、不可能命中 GT 的 sentinel，以保留全体 Query 分母；原始预测文件不变。空长度分组返回 `null`，不产生非法 JSON NaN。

## 测试

无需视频数据集或 API key：

```bash
python -m pytest -q
```

测试通过 FFmpeg 生成三秒视频，使用明确的测试 captioner 与短生命周期子进程模拟模型输出。覆盖完整链路、Query/GT 不被 Ingest 读取、GT 不被 Query 读取、隔离输入结构、篡改检测、失败不重跑、超时/非法输出、AP 匹配和排名语义。没有 FFmpeg 时媒体集成测试会明确跳过。模拟 runner 只存在于测试，不是正式实验选项。

构建镜像后可执行真实容器检查，测试不调用模型 API：

```bash
VMR_TEST_DOCKER=1 python -m pytest -q tests/test_docker_integration.py
```

与官方源码做随机多 GT / 多预测对照，包括失败和空预测：

```bash
VMR_OFFICIAL_ROOT=/path/to/moment_detr \
  python -m pytest -q tests/test_official_parity.py
```

## 代码入口与参考

| 阶段 | 入口 |
| --- | --- |
| Dataset Adapter | `adapters/base.py`、`adapters/qvhighlights.py`、`adapters/uca.py` |
| Ingest | `harness/ingest.py`、`harness/ingest_all.py`、`harness/vlm.py` |
| Freeze | `harness/freeze.py` |
| Query | `harness/run_query.py`、`harness/run_all_queries.py`、`harness/workspace.py` |
| Agent | `agents/codex.py`、`agents/claude_code.py`、`agents/runner.py` |
| Validate / Eval | `harness/validate.py`、`harness/aggregate.py`、`harness/evaluate.py` |

同样支持 `python -m harness.ingest` 等模块形式。第一版不提供 embedding、ASR、重新抽帧、query-specific Wiki、跨 Query memory 或训练。

实现核对的来源：[Codex 非交互模式](https://developers.openai.com/codex/noninteractive)、[OpenAI 图像输入](https://developers.openai.com/api/docs/guides/images-vision)、[Claude Code CLI](https://code.claude.com/docs/en/cli-reference)、[QVHighlights 官方评测](https://github.com/jayleicn/moment_detr/tree/main/standalone_eval)。
