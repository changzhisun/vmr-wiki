# VMR Wiki

将 QVHighlights 与 UCA-VMR 视频转换成固定的视觉时间线，再让每条 Query 在全新的 Codex / Claude Code 进程中完成 Video Moment Retrieval。实现范围对应 [PLAN.md](PLAN.md) 的六个 MVP milestones。

```text
QVHighlights / UCA-VMR annotations → videos / queries / ground_truth manifests
视频 → 固定采样 → VLM captions → Wiki → SHA256 Freeze
选定 Split 的单 Query + 当前 Wiki → 独立容器 / 新进程 → prediction.json
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
- `ingest.vlm.base_url`：按当前配置选择兼容 Chat Completions 的 VLM `/v1` endpoint。
- `ingest.vlm.api_key_env`、`query.api_key_env`：只填写环境变量名，不填写密钥。
- `query.egress_allowed_hosts`：分别为 Codex / Claude Code 声明允许访问的精确模型 API 主机名；不接受通配符或 IP。
- `sample_interval_sec`、预处理尺寸、prompt、temperature、token 上限和 `max_predictions` 是固定实验变量。

通过环境配置 `OPENAI_API_KEY`（Ingest）、`CODEX_API_KEY`（Codex）或 `ANTHROPIC_API_KEY`（Claude Code）。可以在配置中指定其他变量名。当前 Query adapter 使用 API key，不挂载宿主机登录状态。

配置中的相对目录均相对于配置文件所在目录解析。保存到实验中的配置使用完整路径，不包含环境变量的值。Query 模型仍需填写；值为 `REPLACE_...` 占位符时不会发起请求。

## 1. 转换数据集

准备 QVHighlights 官方各 split 的 JSONL annotation，以及对应的 `.mp4` 文件。输入字段为 `qid`、`vid`、`query`、`duration`；带标签 split 还包含 `relevant_windows`。视频文件必须名为 `<vid>.mp4`，且已经裁成 annotation 中对应的片段；文件内 0 秒就是标注的 0 秒，不能直接提供未裁剪的长视频。

```bash
python adapters/qvhighlights.py \
  --annotations /path/to/highlight_val_release.jsonl \
  --split val \
  --video-root /path/to/qvhighlights/videos \
  --output datasets/qvhighlights
```

生成：

```text
datasets/qvhighlights/
├── videos.jsonl
├── queries.jsonl
├── ground_truth.jsonl
└── dataset.json
```

所有 GT windows 原样保留；`queries.jsonl` 只含 `query_id`、`video_id`、`split`、`query`。同一 split 内重复 ID、非法区间、共享视频身份冲突、部分 Query 有标签而其他 Query 无标签都会报错。无标签 split 可以转换和预测，但禁止本地评测；全数据集均无 GT 时不创建 `ground_truth.jsonl`。已有输出目录不会被覆盖。

转换器写入绝对视频路径；手工编写 manifest 时，相对 `video_path` 以 `videos.jsonl` 所在目录为基准。

### Dataset metadata 与任意 Split

一次转换所有 QVHighlights release 文件，并由 Adapter 发现原始 split 名称：

```bash
python adapters/qvhighlights.py \
  --annotation-dir /path/to/qvhighlights/annotations \
  --video-root /path/to/qvhighlights/videos \
  --output datasets/qvhighlights \
  --default-eval-split val
```

Adapter 从 `highlight_<split>_release.jsonl` 提取完整名称，包括 `val_1`、`val_2`。自定义文件名使用重复的 `--annotations 'SPLIT=PATH'`，例如 `--annotations 'validation=/data/custom.jsonl'`。不要将原始名称改写为其他名称。

`dataset.json` 示例：

```json
{
  "name": "qvhighlights",
  "splits": {
    "train": {"has_ground_truth": true},
    "val": {"has_ground_truth": true},
    "test": {"has_ground_truth": false}
  },
  "default_eval_split": "val",
  "evaluator": "qvhighlights"
}
```

Harness 仅将 split 视为非空、不修改的字符串，通过 metadata 验证。`validation` 不会自动映射到 `val`；unknown split 报错并列出全部合法名称。没有保留的 `all` 名称：`--split all` 仅在数据集确实定义了名为 `all` 的 split 时有效。

所有 manifest 行必须带 `split`。同一个视频被多个 split 引用时，`videos.jsonl` 为每个 `(split, video_id)` 保存一个成员关系行，路径与时长必须一致；一个 split 内 Query ID 唯一，不同 split 可以复用相同 Query ID。

```json
{"video_id":"v1","video_path":"/data/v1.mp4","duration":20.0,"split":"validation"}
{"query_id":"q1","video_id":"v1","split":"validation","query":"When does the person stand up?"}
{"query_id":"q1","video_id":"v1","split":"validation","moments":[{"start_sec":2.0,"end_sec":4.0},{"start_sec":12.0,"end_sec":15.0}]}
```

以上三行分别属于 videos、queries、ground_truth。新 Adapter 只需在 `adapters/<name>.py` 中转换原始 annotation、发现 split 并声明 evaluator；Harness 不解释任何 split 名称。可通过 Adapter 的 `evaluate_predictions` 函数扩展官方评测语义，无需在 Harness 增加 dataset 判断。

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
  --output wiki/qvhighlights/videos/video_001
```

正式数据集建议批量执行并冻结：

```bash
python harness/ingest_all.py --dataset qvhighlights --split val --freeze
```

也可以在检查 Wiki 后单独冻结：

```bash
python harness/freeze.py --dataset qvhighlights --split val
```

Ingest 读取 `dataset.json` 和当前 split 的 video 成员关系，不读取 Query 或 GT；按 `0, interval, 2 × interval, … < duration` 抽帧，每张图独立调用同一个固定 VLM prompt。时间戳表示请求的采样时刻，解码器选取该时刻对应的可解码视频帧；它不是事件的精确边界。图像保持纵横比，并限制最长边，不放大小图像。

每个视频输出：

```text
wiki/qvhighlights/videos/<video_id>/
├── wiki.md
├── frames.jsonl
├── frames/
├── ingest.json
└── frozen.json
```

`ingest.json` 记录媒体 SHA256、配置、FFmpeg 版本和内容哈希；`frozen.json` 覆盖 Markdown、JSONL、**每一张图像**及 Ingest 元数据。每个视频独立冻结，实验的 `experiment.json` 保存所选视频的哈希快照；不再使用阻止新增视频的数据集级 `freeze.json`。

完整 Ingest 再次执行时只核验并复用，不重新 caption。失败的临时输出被清除；API 仅对临时网络错误、限流和服务端错误做有限重试，达到 Token 上限的截断响应会直接失败。并发 Ingest 同一个视频会被锁拒绝。Freeze 后单个视频目录只读，其他 split 仍可在 `videos/` 中新增未处理的视频；跨 split 的共享视频只核验和复用。改变 caption 内容配置或媒体时使用新的 Wiki 根目录。VLM provider、endpoint、认证变量、timeout 和 retry 参数作为 provenance 保留，但不影响 `ingest_content_hash`。Ingest 使用源文件的容器时长进行抽帧，不额外比较媒体时长与 annotation 时长。

批量 Ingest 保留 `--jobs`（默认 4）和 `--verbose`。`--jobs 1` 顺序执行；多个 worker 并发处理当前 split 的不同视频，不会对共享 video_id 重复提交任务。Ingest 与批量 Query 的进度条都会显示已完成数量、平均处理速度和预计剩余时间，结束时显示总耗时。收到 Ctrl-C 时，尚未开始的任务立即取消，运行中的 worker 在当前 FFmpeg/VLM 调用结束后的下一个检查点退出并清理 staging 目录；主进程等待 worker 收敛，不会让后台线程继续发布 Wiki。

## 3. 单 Query / 批量 Query

```bash
python harness/run_query.py \
  --dataset qvhighlights \
  --split val \
  --agent codex \
  --experiment codex_wiki_val \
  --query-id 7803

python harness/run_all_queries.py \
  --dataset qvhighlights \
  --split val \
  --agent codex \
  --experiment codex_wiki_val
```

运行 Claude Code 时用独立实验名，并设置相应模型：

```bash
python harness/run_all_queries.py \
  --dataset qvhighlights \
  --split val \
  --agent claude_code \
  --model YOUR_CLAUDE_MODEL \
  --experiment claude_wiki_val
```

Ingest/Freeze/Query 使用显式 `--split`，也可设置 `dataset.split`；这些阶段不会使用 `default_eval_split` 猜测子集。每个实验只运行所选 split，且不要求其他 split 已 Ingest。

每条 Query 创建唯一临时 workspace，内容严格为：

```text
AGENTS.md
task.json
wiki/wiki.md
wiki/frames.jsonl
wiki/frames/
output/
```

### 不透明标识符

公开 VMR benchmark 就在被测模型的训练数据里，而官方的 `qid`、`vid`（QVHighlights 是 YouTube ID，UCA-VMR 是 UCF-Crimes 文件名）和 split 名是**精确的查表键**。只读挂载和 internal network 能阻止 Agent *读取* GT，但阻止不了它认出标签后凭记忆作答，或者拿这个键去问模型 API。

因此 workspace 里不出现真实标识符。每个实验铸造一次随机 `alias_secret`（存在 `experiment.json`，Agent 读不到），`task.json` 里的三个标识符是 `HMAC(secret, kind:value)` 截断后的令牌：

```json
{
  "query_id": "q4f2a91c0d7b83e15",
  "video_id": "v8c17de40ab926f3d",
  "split": "s15a0d1954e686b4c",
  "query": "When does the man open the refrigerator?",
  "max_predictions": 5
}
```

Agent 照旧原样复制这三个值，Harness 按令牌校验后**翻译回真实 ID 再落盘**，所以 `predictions/*.json`、`aggregate` 和 `evaluate` 完全不接触别名。`run_metadata` 同时记录 `task_query_id` / `task_video_id` 以便追溯。别名按实验隔离：同一条 Query 在两个实验里得到不同令牌，无法跨实验关联；恢复同一实验会复用已铸造的 secret，令牌不会中途改变。承载 workspace 的临时目录也用别名命名，因为容器能通过 `/proc` 读到 bind mount 的宿主路径。

Wiki 也不再标注自己的 video id：`wiki.md` 标题固定为 `# Video`。Query 阶段会拒绝仍然写着 `# Video: <video_id>` 的旧 Wiki。

> **这条措施缩小了通道，但没有关闭它。** `query` 文本本身就是任务输入，无法遮蔽，而公开 benchmark 的 query 文本同样可检索。采样帧里若出现视频自带的标题字幕也会泄露。把别名理解为「移除了精确查表键」，不要当成去污染的保证；报告结果时应当说明这一点。

### 旧 Wiki 迁移

`render_wiki` 过去会写 `# Video: <video_id>`。captions 和采样帧完全不受影响，所以不必重新调用 VLM，只改标题即可：

```bash
python harness/migrate_wiki_title.py --dataset uca            # 先干跑，报告将要改动的数量
python harness/migrate_wiki_title.py --dataset uca --apply
```

迁移前会核验除 `wiki.md` 外的每一项 `content_hashes` 仍然吻合（这就是 captions 未被改动的证据），然后重写标题、更新记录的哈希，并在 `ingest.json` 的 `migrations` 中留痕。已 Freeze 的 Wiki 会被拒绝：先删掉 `frozen.json`，迁移后重新 Freeze，让 seal 证明真实存在的内容。

`task.json` 保留 `query_id`、`video_id`、`split`、`query` 和固定的 `max_predictions`（均为别名形式）。Agent 必须在预测中原样复制 split；示例中为可读性使用真实 ID：

```json
{
  "query_id": "q1",
  "video_id": "v1",
  "split": "validation",
  "moments": [
    {"start_sec": 2.0, "end_sec": 4.0, "score": 0.91},
    {"start_sec": 12.0, "end_sec": 15.0, "score": 0.42}
  ],
  "evidence": "Visual evidence from the frozen timeline."
}
```

`moments` 按 score 降序排列，可为空。每个 moment 必含 start/end/score；顶层和 moment 内的 `evidence` 都是可选字符串。不同 split 的实验必须使用不同 experiment 名称。

宿主仓库、视频、GT、其他 Query、其他预测都不挂载进容器。容器以非 root 用户运行，根文件系统和整个 workspace 只读，唯一的任务输出挂载点 `output/` 可写。容器 Home 和临时目录每次新建且退出销毁，CLI 禁用 session 持久化，不使用 resume。Claude 显式加载同一份 `AGENTS.md`。

Agent 容器只连接每次运行新建的 Docker internal network，没有直接公网路由。另一个不持有 API key、也不挂载 workspace 的最小代理 sidecar 同时连接 internal network 和 Docker bridge，仅允许 HTTPS CONNECT 到 `query.egress_allowed_hosts` 中的精确主机名和 443 端口；运行结束后 Agent、代理和网络都会被删除。模板仍明确禁止联网检索，Claude 仅开放文件及 shell 内置工具，不启用额外 MCP。修改 allowlist 会改变实验配置哈希，应使用新实验名。

Harness 等待进程结束、验证输入未变、校验输出，再保存结果并清除 workspace。超时会强制删除整个容器和进程。JSON 缺失、解析失败、字段多余/缺失、错误 ID、布尔值冒充数字、NaN、时间越界、分数越界、排序错误、预测数量超限、额外输出文件等均记录为 `invalid_output` 失败，**不会自动修复或重新调用 Agent**。`moments: []` 是成功的 abstention。

每次运行都记录 `failure_kind`，把"Agent 没做好"和"Harness / 基础设施坏了"分开：

| failure_kind | 含义 | 归属 | 是否重跑 |
| --- | --- | --- | --- |
| `timeout` | Agent 超时 | Agent | 否，终局 |
| `agent_error` | Agent 进程非零退出 | Agent | 否，终局 |
| `invalid_output` | 输出缺失、无法解析、schema 不合法、多余产物 | Agent | 否，终局 |
| `tampered` | Agent 改动了冻结输入 | Agent | 否，终局 |
| `harness_error` | Docker 不可用、磁盘错误、Harness 不变量被破坏 | Harness | 是 |
| `interrupted` | Ctrl-C 或进程未收尾 | Harness | 是 |

Agent 自己的失败是终局，绝不会为同一条 Query 再启动第二个进程，因此不存在 best-of-N。Harness 侧失败不是 Agent 的成绩：`run_all_queries.py` **立即中止整批**而不是把剩余 Query 记成零分，修好原因后重跑会自动重试这些 Query，并在 `attempts` 和 `superseded_failures` 中留下审计痕迹，重跑过的 Query 不会被误认为首次尝试。成功记录和 Agent 失败记录都不会被重跑。没有 `failure_kind` 字段的历史记录按终局处理。配置、模型、Agent、代码、模板、manifest 或镜像 ID 变化时，必须使用新实验名。

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

元数据包含 dataset、split、agent、时间、exit code、失败原因、Wiki/config/template/source/manifest 哈希、镜像 ID 和 Git commit；目录不是 Git 仓库时 commit 为 `null`，代码仍有内容哈希。

若宿主进程被 `SIGKILL` 或机器断电，正常的 `finally` 清理无法执行。确认没有相关进程/容器继续运行后，可人工移除对应 `.experiment.lock`、`.ingest.lock` 和残留临时目录；不要删除已记录的 Query 结果来假装首次执行。

## 4. 聚合与评测

批量运行因部分 Query 的 **Agent 侧**失败而返回非零 exit code 时，应继续聚合与评测——这些失败是有效的零分。但若批量运行是因 Harness 侧失败而**中止**（输出中带 `harness failure after N of M queries`），应先修好原因重跑，否则评测会拒绝这批结果。

```bash
python harness/aggregate.py \
  --dataset qvhighlights \
  --split val \
  --input results/codex_wiki_val/predictions \
  --output results/codex_wiki_val/predictions.jsonl

python harness/evaluate.py \
  --dataset qvhighlights \
  --split val \
  --pred results/codex_wiki_val/predictions.jsonl
```

聚合会验证所有预测的 schema、split、Query ID、video 映射，以及实验/逐 Query 元数据中的 dataset 身份；不同 split 不会被筛掉后静默评测，而是明确拒绝。独立预测目录必须通过配置指定 dataset 和 split，以查询 manifest 核验归属。输出保留全部 moments 和 evidence，并生成 `predictions.jsonl.metadata.json`，记录 dataset、split、annotation 哈希及聚合文件 SHA256。移动结果时一起保留这个 provenance 文件；评测默认强制要求 sidecar，缺失时不会静默降级。确需评测未经 `aggregate.py` 生成的外部原始 submission 时，必须显式传入 `--allow-unverified-predictions`，同时指定正确的 dataset/split。

评测首先读取 `dataset.json` 并检查 `has_ground_truth`。若为 false，在打开 GT 或预测文件之前就拒绝本地评测，提示该 split 只能生成 prediction。为 true 时只选择该 split 的 GT；其他 split 的 GT 不计入分母。

未显式指定 `--split` 时，聚合使用保存的实验 split。评测按保存配置/指定配置的 `dataset.split`，再按可选 `default_eval_split` 选择；缺少选择依据时报错。`default_eval_split` 必须声明且有 GT。可用 `--gt` 显式指定同一数据集目录内的 GT 文件，但它不能绕过 metadata 的无 GT 限制。

默认写入相邻的 `metrics.json`；可通过 `--output` 指定位置。评测读取实验保存的配置和 run metadata；数据集元数据选择 QVHighlights evaluator，或显式使用 `--evaluator generic`。UCA-VMR 的 `dataset.json` 中 `evaluator` 为 `generic`，因此直接使用通用 evaluator，无需 `--evaluator` 参数。

**所有指标的单位为百分数，范围 0–100。** 分母是所选 split 的全部带标签 Query，缺失/失败结果按零命中处理；不因某个 Query 失败就从评测集删除它。只跑一个 Query 时评测该 split，其余同 split 内未运行的 Query 也会计入失败；需要子集实验时应先准备对应子集的 manifests。

若 run metadata 中存在 `harness_error` 或 `interrupted`，评测**直接拒绝**并列出对应 Query：这些 Query 根本没有测量值，把它们当成零分会让一次 Docker 故障看起来像 Agent 不会做 VMR。修好原因后重跑即可（这两类会自动重试）；确实要按零分计入时显式传 `--allow-harness-failures`。

通用 evaluator 输出配置的 `R@K,IoU=T`，语义为前 K 个预测命中任意一个 GT 即成功。UCA-VMR 使用此 evaluator（每个 Query 只有一个 GT moment，属于 `any_acceptable_moment` 语义）。QVHighlights adapter 额外输出：

- `MR-full-mAP`：主指标，IoU 0.50 到 0.95、间隔 0.05 的平均 AP。
- `MR-full-mAP@0.5`、`MR-full-mAP@0.75`，以及 R1@0.5 / R1@0.7。
- short `(0,10]`、middle `(10,30]`、long `(30,150]` 秒的分组 mAP。

QVHighlights AP 使用前 10 个预测，按置信度逐一与尚未匹配的 GT 做匹配；重复命中同一个 GT 不能重复加分。实现遵循官方 moment retrieval 语义，不计算与本任务无关的 highlight/saliency 指标。没有 Query 的长度组返回 JSON `null`。

输出同时包括 `failed_runs`、失败 ID、缺失数量、abstention 数量、每条 Query 平均预测数、top K 和 IoU thresholds。提供 run metadata 时额外输出 `run_failures`：按 `failure_kind` 分类计数，外加 `unattempted`（从未启动）和 `unclassified`（历史记录）。聚合保留所有预测和 evidence，不会静默降成 top-1，也不会重排或修复不合法结果。

当存在未作答的 Query 时，metrics.json 追加一个 `successful_only` 块，用**同一套指标**重算仅覆盖已作答 Query 的分数。顶层数字始终以整个 split 为分母，是对外汇报的口径；`successful_only` 用来判断差距来自检索质量还是来自格式遵从率与覆盖率。对比 Codex 与 Claude Code 时必须同时看这两个数——否则 JSON 合规性的差异会被读成 VMR 能力的差异。

```json
{
  "num_queries": 6,
  "failed_runs": 2,
  "run_failures": {"timeout": 1, "invalid_output": 1, "harness_error": 0,
                   "interrupted": 0, "agent_error": 0, "tampered": 0,
                   "unclassified": 0, "unattempted": 0},
  "primary_metric": "MR-full-mAP",
  "primary_score": 33.33,
  "successful_only": {"num_queries": 4, "primary_score": 50.0}
}
```

### 使用官方代码

优先使用官方 evaluator 时，提供本地 [moment_detr](https://github.com/jayleicn/moment_detr) checkout：

```bash
python -m pip install -e '.[official]'
python harness/evaluate.py \
  --dataset qvhighlights \
  --split val \
  --pred results/codex_wiki_val/predictions.jsonl \
  --official-root /path/to/moment_detr
```

此路径在独立 evaluator 进程中调用官方 `compute_mr_ap` / `compute_mr_r1`，记录官方源码 SHA256；官方执行失败会明确报错，不静默降级。未指定官方 checkout 时使用本地 dataset adapter，指标中标注 `implementation`。

上游 R1 不能处理空预测，AP 会遗漏空列表。因此官方桥接仅在评测器内将缺失/空预测表示成零长度、零分数、不可能命中 GT 的 sentinel，以保留全体 Query 分母；原始预测文件不变。空长度分组返回 `null`，不产生非法 JSON NaN。

## 旧数据迁移（方案 A）

旧 manifest 缺少 split 或没有 `dataset.json` 时，程序明确提示重新运行 Dataset Adapter；不自动补 `default`，也不猜测 train/val/test。请在新目录重新转换后切换配置，保留原始 annotation 名称。旧实验和不含 split 的预测不能与新实验混用。

Wiki 不包含 annotation split；原有有效 Video Wiki 可由用户迁移至 `wiki/<dataset>/videos/<video_id>/`，保持目录内容及哈希不变后重新核验。无需为了新 split 重新 caption。旧数据集根目录若被置为只读，需要为新的 `videos/` 布局准备可写父目录；本实现不自动移动或删除旧产物。

## 测试

无需视频数据集或 API key：

```bash
python -m pytest -q
```

Split 测试覆盖任意名称、unknown split、严格布尔类型、默认 eval split、视频/Query 筛选、共享 Wiki 增量冻结、跨 split Query ID、预测隔离、无 GT 拒绝和混合 provenance。

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
| Split / Dataset metadata | `harness/dataset.py`、`harness/results.py` |
| Dataset Adapter | `adapters/base.py`、`adapters/qvhighlights.py`、`adapters/uca.py` |
| Ingest | `harness/ingest.py`、`harness/ingest_all.py`、`harness/vlm.py` |
| Freeze | `harness/freeze.py` |
| Query | `harness/run_query.py`、`harness/run_all_queries.py`、`harness/workspace.py` |
| Agent | `agents/codex.py`、`agents/claude_code.py`、`agents/runner.py` |
| Validate / Eval | `harness/validate.py`、`harness/aggregate.py`、`harness/evaluate.py` |

同样支持 `python -m harness.ingest` 等模块形式。第一版不提供 embedding、ASR、重新抽帧、query-specific Wiki、跨 Query memory 或训练。

实现核对的来源：[Codex 非交互模式](https://developers.openai.com/codex/noninteractive)、[OpenAI 图像输入](https://developers.openai.com/api/docs/guides/images-vision)、[Claude Code CLI](https://code.claude.com/docs/en/cli-reference)、[QVHighlights 官方评测](https://github.com/jayleicn/moment_detr/tree/main/standalone_eval)。
