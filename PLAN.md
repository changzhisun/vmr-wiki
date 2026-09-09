# VMR Wiki 实现计划

当前版本包含通用 Dataset Split 支持；Split 是 Adapter 定义的 opaque string，Harness 仅验证、筛选和记录。旧的无 split manifest 必须重新运行 Adapter。

## 1. 项目目标

将一个现成的 Video Moment Retrieval（VMR）数据集改造成一个适合 Codex / Claude Code 等 LLM Agent 独立求解的标准化 Harness。

核心目标：

> 对每个视频先进行一次固定的离线 Ingest，生成不可变的 Video Wiki；之后每个 Query 都在完全独立、结构一致的上下文环境中运行一个新的非交互式 Agent，输出统一格式的 JSON 预测文件；最后聚合所有 Query 的结果，并使用数据集对应的 VMR evaluator 统一评测。

第一版重点不是设计新的 VMR 模型，而是建立一个：

- 可复现；
- Query 间相互隔离；
- Wiki 格式固定；
- 一个 Query 可对应多个 Ground-Truth moments；
- 一个 Query 可输出多个预测 moments；
- 预测 moments 有固定排序与分数格式；
- 可适配现有 VMR 数据集；
- 可比较 Codex / Claude Code 等不同 Agent；

的实验 Harness。

---

## 2. 总体架构

整体流程分成三个阶段：

```text
              Dataset Preparation

原始 VMR Dataset
        ↓
Dataset Adapter
        ↓
统一 videos / queries / ground truth


                  Ingest

video
  ↓
固定采样
  ↓
固定 VLM API
  ↓
timestamped frame captions
  ↓
Video Wiki
  ↓
FREEZE


                  Query

one query
   +
frozen video wiki
   +
fixed instructions
        ↓
fresh Codex / Claude Code process
        ↓
prediction.json
        ↓
process exits


                Evaluation

所有 prediction.json
        ↓
aggregate
        ↓
标准 / 官方 VMR evaluator
        ↓
metrics.json
```

---

## 3. 核心原则

### 3.1 Ingest 与 Query 完全分离

Ingest 是普通 Python 脚本。

它不使用 Codex / Claude Code Agent。

它只负责：

```text
video
→ frames
→ VLM captions
→ wiki
```

Query 阶段才使用 Agent。

### 3.2 一个视频只 Ingest 一次

同一个视频的所有 Query 必须使用完全相同的 Wiki。

```text
video_001
    ↓
 ingest once
    ↓
 wiki_001
    ↓
   FREEZE
    │
    ├── q001
    ├── q002
    └── q003
```

Query 阶段不允许修改 Wiki。

### 3.3 每个 Query 独立运行

每个 Query：

- 启动一个新的 Codex / Claude Code 非交互进程；
- 使用一个新的临时 workspace；
- 不继承任何历史对话；
- 不读取其他 Query；
- 不读取其他 Prediction；
- 不读取 Ground Truth；
- 完成后进程退出；
- workspace 删除。

### 3.3.1 Workspace 不得包含可识别的数据集标识符

隔离必须同时覆盖「读得到的字节」和「认得出的标签」。公开 benchmark 在被测模型的权重里，官方 `qid` / `vid` / split 名是精确查表键，所以只读挂载和 internal network 并不足够。

```text
真实 qid / vid / split
        ↓  HMAC(per-experiment secret)
不透明令牌  →  task.json / workspace 目录名 / wiki.md 标题
        ↓  Harness 校验后翻译回真实 ID
predictions / aggregate / evaluate
```

残余通道必须如实声明：`query` 文本本身是任务输入，无法遮蔽，公开数据集的 query 文本同样可检索。别名只移除精确查表键，不构成去污染保证。

### 3.4 Wiki 与 Query 无关

Wiki 只能由以下信息生成：

```text
video
+
固定采样策略
+
固定 VLM
+
固定 VLM prompt
```

不能：

```text
video + query → wiki
```

也不能读取 Query 对应的 Ground Truth moments。

### 3.5 一个 Query 可以对应多个 Moment

Harness 从一开始就假设：

```text
一个 Query
→ 可能有多个 GT moments
→ Agent 也可以返回多个 prediction moments
```

例如 Query：

```text
When does the person open the refrigerator?
```

视频中可能发生三次：

```text
[12.0, 15.5]
[48.2, 51.0]
[103.4, 106.8]
```

因此数据结构不能假设每个 Query 只有唯一的 `[start, end]`。

---

## 4. Repo 目录结构

Repo 名称固定为：

```text
vmr-wiki
```

第一版建议：

```text
vmr-wiki/
│
├── README.md
├── PLAN.md
├── config.yaml
│
├── harness/
│   ├── ingest.py
│   ├── ingest_all.py
│   ├── run_query.py
│   ├── run_all_queries.py
│   ├── workspace.py
│   ├── validate.py
│   ├── aggregate.py
│   └── evaluate.py
│
├── adapters/
│   ├── base.py
│   └── <dataset_name>.py
│
├── agents/
│   ├── codex.py
│   └── claude_code.py
│
├── templates/
│   ├── AGENTS.md
│   └── query_prompt.md
│
├── datasets/
│   └── <dataset_name>/
│       ├── dataset.json
│       ├── videos.jsonl
│       ├── queries.jsonl
│       └── ground_truth.jsonl
│
├── videos/
│   └── <dataset_name>/
│       └── ...
│
├── wiki/
│   └── <dataset_name>/
│       └── videos/
│           └── <video_id>/
│               ├── wiki.md
│               ├── frames.jsonl
│               └── frames/
│
├── runs/
│
└── results/
    └── <experiment_name>/
```

第一版不提供额外 Agent tools。

Query Agent 只使用 Harness 预先提供的：

```text
task.json
wiki.md
frames.jsonl
frames/
```

因此 Ingest 阶段需要提供足够用于 VMR 的 timestamped visual representation。

---

## 5. Dataset Adapter 与 Split

Adapter 定义并保留原始 split 名称，不能将 `validation` 改名为 `val`。核心 Harness 不包含任何数据集名称或 split 名称的条件分支。

标准化输出：

```text
datasets/<dataset>/
├── dataset.json
├── videos.jsonl
├── queries.jsonl
└── ground_truth.jsonl  # 仅含公开标签；全数据集无 GT 时不生成
```

`dataset.json`：

```json
{
  "name": "example_dataset",
  "splits": {
    "training": {"has_ground_truth": true},
    "validation": {"has_ground_truth": true},
    "testing": {"has_ground_truth": false}
  },
  "default_eval_split": "validation",
  "evaluator": "generic"
}
```

Split 可以是任意非空字符串。`has_ground_truth` 必须为 JSON boolean，`default_eval_split` 可选，但提供时必须是已声明且有 GT 的 split。Unknown split 报错并列出合法名称，不做别名猜测。没有保留的 `all` split 名称；一次运行选择一个确切 split。

### 5.1 videos.jsonl

```json
{"video_id":"video_001","video_path":"videos/video_001.mp4","duration":632.4,"split":"validation"}
```

同一 video_id 可以在多个 split 中各有一行，表示 annotation 成员关系；其 video_path 和 duration 必须一致。同一 `(split, video_id)` 不可重复。Wiki 仍然只生成一份。

### 5.2 queries.jsonl

```json
{"query_id":"q001","video_id":"video_001","split":"validation","query":"When does the man open the refrigerator?"}
```

Query ID 在 split 内唯一；不同 split 允许复用 ID。Query 必须引用该 split 中的视频，且不能包含 GT 时间信息。

### 5.3 ground_truth.jsonl

```json
{
  "query_id": "q001",
  "video_id": "video_001",
  "split": "validation",
  "moments": [
    {"start_sec": 12.0, "end_sec": 15.5},
    {"start_sec": 48.2, "end_sec": 51.0},
    {"start_sec": 103.4, "end_sec": 106.8}
  ]
}
```

单 GT 同样使用 moments 数组。有 GT 的 split 必须覆盖其全部 Query；不允许部分 Query 有标签、部分没有。无 GT 的 split 可以 Ingest、Query 和 Aggregate，Evaluator 必须在读取 GT 前拒绝本地评测。

QVHighlights Adapter 可以发现 `highlight_<split>_release.jsonl`，或接受显式 `SPLIT=PATH` 映射。UCA Adapter 使用自身文件命名规则。所有数据集专属的发现、转换、评测逻辑限制在 `adapters/`。

---

## 6. Ingest 阶段

### 6.1 接口

单视频：

```bash
python harness/ingest.py \
  --video /path/to/video.mp4 \
  --video-id video_001 \
  --output wiki/<dataset>/videos/video_001
```

批量：

```bash
python harness/ingest_all.py \
  --dataset <dataset_name> \
  --split <split_name>
```

### 6.2 固定流程

每个视频严格执行同一个流程：

```text
读取视频 metadata
        ↓
固定时间间隔采样 frame
        ↓
对每张 frame 调用固定 VLM API
        ↓
获得 timestamp + caption
        ↓
写 frames.jsonl
        ↓
生成 wiki.md
```

所有视频必须使用相同：

- sampling interval；
- VLM model；
- VLM prompt；
- image preprocessing；
- temperature；
- max tokens；
- output schema。

---

## 7. Wiki 格式

每个视频固定生成：

```text
wiki/<dataset>/videos/<video_id>/
├── wiki.md
├── frames.jsonl
└── frames/
```

### 7.1 frames.jsonl

机器可读格式：

```json
{"frame_id":"f000001","timestamp":0.0,"frame":"frames/000001.jpg","caption":"A man enters a kitchen."}
{"frame_id":"f000002","timestamp":5.0,"frame":"frames/000002.jpg","caption":"The man walks toward a refrigerator."}
{"frame_id":"f000003","timestamp":10.0,"frame":"frames/000003.jpg","caption":"The refrigerator door is open."}
```

每一条必须包含：

```text
frame_id
timestamp
frame
caption
```

### 7.2 wiki.md

给 Agent 阅读的统一 Markdown：

```markdown
# Video: video_001

## Metadata

- Duration: 632.4 seconds
- Sampling interval: 5.0 seconds
- Number of frames: 127

## Timeline

### 0.0s

A man enters a kitchen.

Frame: `frames/000001.jpg`

### 5.0s

The man walks toward a refrigerator.

Frame: `frames/000002.jpg`

### 10.0s

The refrigerator door is open.

Frame: `frames/000003.jpg`
```

MVP 不要求构造：

- concept pages；
- event pages；
- knowledge graph；
- embedding index；
- summary hierarchy。

Wiki 就是一个固定格式的 timestamped visual timeline。

---

## 8. Wiki Freeze

Wiki 的物理布局固定为：

```text
wiki/<dataset>/videos/<video_id>/
├── wiki.md
├── frames.jsonl
├── frames/
├── ingest.json
└── frozen.json
```

`freeze.py --dataset DATASET --split SPLIT` 只核验并冻结选定 split 的 unique videos。单个视频目录只读；父目录不整体锁死，因此后续 split 可以添加未处理的视频。

共享 video_id 复用已有 Wiki，不重复 caption。每个 frozen.json 覆盖 Markdown、JSONL、全部图像及 Ingest 元数据的 SHA256。实验 metadata 保存当前 split 使用的 `video_id -> wiki_hash` 快照；Query 前后均验证输入，不能修改已冻结内容。

Ingest 配置中决定 caption 内容的参数变化时应使用新 Wiki 根目录。VLM provider、endpoint、认证变量、timeout 和 retry 次数等传输参数保留在 provenance 中，但不参与 ingest_content_hash，也不触发重复 caption。不存在 split 专属 Wiki 目录，也不使用阻止其他 split 增量 Ingest 的全局 freeze.json。

---

## 9. Query Harness

对于：

```json
{
  "query_id": "q001",
  "video_id": "video_001",
  "split": "val",
  "query": "When does the man open the refrigerator?"
}
```

Harness 创建一个全新的临时目录：

```text
runs/q001/
├── AGENTS.md
├── task.json
├── wiki/
│   ├── wiki.md
│   ├── frames.jsonl
│   └── frames/
└── output/
```

第一版不提供：

```text
video.mp4
ground_truth
其他 query
其他 prediction
其他 video wiki
额外 tools
```

Query Agent 只能基于当前视频已经冻结的 Wiki 和 sampled frames 完成任务。

这样可以最大程度保证 Query 隔离。

---

## 10. task.json

统一格式：

```json
{
  "query_id": "q001",
  "video_id": "video_001",
  "split": "val",
  "query": "When does the man open the refrigerator?",
  "max_predictions": 5
}
```

`max_predictions` 用于固定每个 Agent 最多允许返回多少个 moment。

这个值属于实验配置，不应由 Agent 自己决定。

每个 workspace 中只存在一个 Query。

---

## 11. Agent 指令

`templates/AGENTS.md` 对所有 Query 完全相同。

建议：

```markdown
# VMR Wiki Harness

你正在独立完成一个 Video Moment Retrieval 任务。

## 输入

读取：

- `task.json`
- `wiki/wiki.md`
- `wiki/frames.jsonl`
- `wiki/frames/`

Wiki 是当前视频预先生成并冻结的 timestamped visual timeline。

## 任务

根据 `task.json` 中的自然语言 Query，在 Wiki 中找到一个或多个最匹配的 temporal moments。

你可以：

1. 阅读 `wiki.md`；
2. 搜索相关 caption；
3. 根据 timestamp 判断候选时间段；
4. 查看相关 sampled frames；
5. 返回按置信度从高到低排序的 moment 列表。

如果视频中同一个 Query 对应多个独立的相关片段，可以返回多个 moments。

预测数量不能超过 `task.json` 中的 `max_predictions`。

## 限制

不要修改 Wiki。

不要假设存在任何 Ground Truth。

不要读取当前 workspace 之外的文件。

## 输出

必须创建：

`output/prediction.json`

并严格遵守指定 JSON schema。
```

所有实验始终使用同一份模板。

---

## 12. Query 输出格式

每条 Query 只生成 `output/prediction.json`，其中可包含多个按 score 降序排列的 moments：

```json
{
  "query_id": "q001",
  "video_id": "video_001",
  "split": "val",
  "moments": [
    {"start_sec": 12.0, "end_sec": 15.0, "score": 0.92},
    {"start_sec": 48.0, "end_sec": 51.5, "score": 0.81}
  ],
  "evidence": "Visual evidence from the frozen timeline."
}
```

query_id、video_id、split 必须与当前 task 一致。evidence 可放在顶层或每个 moment 内，为可选字符串，不参与指标计算。聚合保留全部 moments、split 和已提供的 evidence。

## 13. Prediction JSON Schema 约束

必需字段：query_id、video_id、split、moments。每个 moment 必须包含 start_sec、end_sec、score，不接受未定义字段；evidence 可选。

```text
0 <= start_sec < end_sec <= video duration
0 <= score <= 1
len(moments) <= max_predictions
moments[i].score >= moments[i+1].score
```

数值必须有限且不是布尔值；允许 `moments: []` 表示成功 abstention。输出缺失、JSON 无法解析、字段缺失、额外输出、错误 split/ID、数量越限、时间非法或排序错误均记为 failed run，不自动修复或重跑 Agent。

### 13.1 Failed Run 必须区分归属

failed run 分两类，绝不能折叠成同一个数字：

```text
Agent 侧（有效的零分，终局，不重跑）
  timeout / agent_error / invalid_output / tampered

Harness 侧（没有测量值，自动重试）
  harness_error / interrupted
```

一次 Docker 故障若被记成零分，会让"Agent 不会做 VMR"和"环境坏了"无法区分。因此：批量运行遇到 Harness 侧失败立即中止；评测遇到 Harness 侧失败直接拒绝，除非显式 opt-in。

`metrics.json` 顶层数字以整个 split 为分母（对外口径），并追加 `successful_only` 用同一套指标重算仅覆盖已作答 Query 的分数。比较不同 Agent 时必须同时看两个数，否则 JSON 合规性差异会被误读成检索能力差异。

---

## 14. Agent Runner

统一接口：

```bash
python harness/run_all_queries.py \
  --dataset <dataset_name> \
  --split <split_name> \
  --agent codex \
  --experiment codex_wiki_v1
```

内部逻辑：

```text
for query in selected_split_queries:

    创建 fresh workspace

    写 task.json

    复制固定 AGENTS.md

    复制 / mount 当前 video 的 frozen wiki

    创建空 output/

    启动全新的 Agent 非交互进程

    等待进程退出

    validate prediction.json

    保存结果

    删除 workspace
```

Codex：

```text
one query
→ one fresh codex process
→ process exits
```

Claude Code：

```text
one query
→ one fresh claude process
→ process exits
```

不使用历史 session。

不 resume。

---

## 15. Results 目录

例如：

```text
results/
└── codex_wiki_v1/
    ├── predictions/
    │   ├── q001.json
    │   ├── q002.json
    │   └── ...
    │
    ├── run_metadata/
    │   ├── q001.json
    │   └── ...
    │
    └── config.yaml
```

Query prediction 与运行 metadata 分开保存。

---

## 16. Run Metadata

建议每个 Query 同时由 Harness 记录：

```json
{
  "query_id": "q001",
  "dataset": "qvhighlights",
  "split": "val",
  "agent": "codex",
  "model": "MODEL_NAME",
  "exit_code": 0,
  "wiki_hash": "...",
  "config_hash": "...",
  "git_commit": "...",
  "started_at": "...",
  "finished_at": "..."
}
```

这些数据不由 Agent 生成，而由外部 Harness 记录。

---

## 17. Aggregate

当所有 Query 完成：

```bash
python harness/aggregate.py \
  --dataset <dataset_name> \
  --split <split_name> \
  --input results/codex_wiki_v1/predictions \
  --output results/codex_wiki_v1/predictions.jsonl
```

聚合后仍然保留多个 moments：

```json
{"query_id":"q001","video_id":"video_001","split":"val","moments":[{"start_sec":12.0,"end_sec":15.0,"score":0.92},{"start_sec":48.0,"end_sec":51.5,"score":0.81}]}
{"query_id":"q002","video_id":"video_001","split":"val","moments":[{"start_sec":73.0,"end_sec":76.0,"score":0.88}]}
```

Aggregate 阶段不能只保留 top-1，除非某个具体 evaluator 明确要求 top-1。

---

## 18. Evaluation

Evaluator 完全独立于 Agent。

执行：

```bash
python harness/evaluate.py \
  --dataset <dataset_name> \
  --split <split_name> \
  --pred results/codex_wiki_v1/predictions.jsonl \
  --gt datasets/<dataset>/ground_truth.jsonl
```

### 18.1 优先使用数据集官方评测

不同 VMR 数据集对“多个 GT / 多个预测”的定义可能不同。

因此优先级是：

```text
1. 原数据集官方 evaluator
2. Dataset Adapter 中实现与官方语义一致的 evaluator
3. 最后才使用 Harness 的通用 evaluator
```

Harness 不应强行把所有数据集压缩成单一 top-1 指标。

### 18.2 通用 Retrieval 指标

如果数据集没有特殊官方定义，可以使用 ranked moment retrieval：

```text
Recall@K, IoU=T
```

对于一个 Query：

- 取预测列表前 K 个 moments；
- 只要其中任意一个 prediction 与任意一个 GT moment 的 temporal IoU >= T；
- 则该 Query 在该指标下视为 hit。

例如：

```text
R@1, IoU=0.5
R@5, IoU=0.5
R@1, IoU=0.7
R@5, IoU=0.7
```

如果数据集将多个 GT moments 视为多个独立 relevant instances，也可以额外计算：

```text
GT moment coverage
mAP
```

但这些指标应由 dataset adapter 根据原 benchmark 的定义决定。

### 18.3 MVP 至少记录

```text
官方主指标
R@1 / R@K（如果适用）
IoU thresholds
Failed Runs（并按 failure_kind 分类，见 13.1）
Unattempted queries
Average number of predictions per query
successful_only：仅已作答 Query 的同套指标
```

---

## 19. 多 Moment 的评测注意事项

必须区分两种情况。

### 情况 A：多个 GT 表示多个都可接受的答案

例如三个时间段都可以正确回答同一个 Query。

这时常见语义是：

```text
prediction 命中任意一个 GT 即可
```

### 情况 B：多个 GT 表示视频中存在多个 relevant moments

例如 Query 要求检索所有相关片段。

这时需要考虑：

```text
多个 predictions 是否覆盖多个 GT moments
```

可能使用：

```text
Recall@K
mAP
moment-level recall
```

Dataset Adapter 必须保留原数据集的语义，不能在转换时丢失这个区别。

---

## 20. Codex / Claude Code 对比

Harness 层提供统一 Agent Adapter：

```text
agents/
├── codex.py
└── claude_code.py
```

它们只负责：

```text
workspace path
+
fixed prompt
↓
启动非交互 Agent
↓
等待结束
↓
返回 exit code
```

不同 Agent 必须分别保存实验：

```text
results/codex_wiki_v1/
results/claude_wiki_v1/
```

不能在同一个实验里混用不同 Agent。

---

## 21. config.yaml

所有实验变量集中管理。

例如：

```yaml
dataset:
  name: qvhighlights
  split: val  # exact name from dataset.json; may be overridden by --split

ingest:
  sample_interval_sec: 5.0

  vlm:
    provider: xxx
    model: xxx
    temperature: 0
    max_tokens: 256

query:
  agent: codex
  model: xxx
  max_predictions: 5

evaluation:
  top_k:
    - 1
    - 5
  iou_thresholds:
    - 0.3
    - 0.5
    - 0.7
```

`max_predictions` 必须固定，避免不同 Agent 因为返回数量不同而产生不公平比较。

正式实验时保存完整 config。

---

## 22. MVP 开发顺序

### Milestone 1：Dataset Adapter

完成一个现成 VMR 数据集的转换：

```text
raw annotations
↓
videos.jsonl
queries.jsonl
ground_truth.jsonl
```

重点确保：

```text
一个 query 的多个 GT moments 不会在转换时丢失
```

只支持一个 Dataset 即可。

### Milestone 2：Ingest

完成：

```text
video
↓
fixed frame sampling
↓
VLM API
↓
frames.jsonl
↓
wiki.md
```

先测试少量视频。

### Milestone 3：Freeze Wiki

完成所有测试视频 Ingest 后锁定：

```text
wiki/
```

并记录 hash。

### Milestone 4：单 Query Harness

选择一个 Query。

自动创建：

```text
fresh workspace
```

启动一次：

```text
Codex / Claude Code non-interactive
```

成功生成包含：

```text
moments: [...]
```

的：

```text
output/prediction.json
```

即完成最小闭环。

### Milestone 5：批量 Query

实现：

```text
run_all_queries.py
```

保证：

```text
每个 Query
=
新的 workspace
+
新的 Agent process
```

### Milestone 6：Aggregate + Eval

完成：

```text
individual JSON
↓
predictions.jsonl
↓
official / dataset-specific evaluator
↓
metrics.json
```

并确保整个链路支持：

```text
multiple GT moments
multiple predicted moments
```

---

## 23. MVP 明确不做

第一版暂时不实现：

```text
Agent-side tools
局部重新抽帧
Embedding
FAISS
Vector DB
BM25
ASR
Whisper
Event Graph
Knowledge Graph
Concept Wiki
Query-specific Wiki
Query 后更新 Wiki
模型训练
Fine-tuning
多轮 Agent session
跨 Query Memory
```

这些功能之后都可以作为独立变量逐步加入。

---

## 24. MVP 最终定义

第一版系统可以概括为：

```text
          STANDARD VMR DATASET
                   ↓
             Dataset Adapter
                   ↓
      ┌────────────┴────────────┐
      ↓                         ↓
    Video                     Query
      ↓                         │
fixed offline ingest            │
      ↓                         │
timestamped Video Wiki          │
      ↓                         │
    FREEZE                      │
      │                         │
      └────────────┬────────────┘
                   ↓
          isolated workspace
                   ↓
        fresh coding agent
                   ↓
       prediction.json
       moments: [...] 
                   ↓
              aggregate
                   ↓
            standard eval
```

核心实验单位是：

> **一个 Query = 一个新的、完全独立的 Agent 进程 + 一个只包含当前 Query 和当前 Video Wiki 的标准化 workspace。**

核心 Wiki 单位是：

> **一个 Video = 一份与 Query 无关、由固定 VLM Ingest Pipeline 一次性生成并冻结的 timestamped visual Wiki。**

核心 Moment 接口是：

> **Ground Truth 和 Prediction 都统一使用 `moments: [...]` 数组，因此天然支持一个 Query 对应一个或多个 temporal moments。**

这就是第一版 `vmr-wiki` Harness。

---

## 25. Split 实现与验收

已实现统一入口：

```bash
python harness/ingest_all.py --dataset DATASET --split SPLIT --freeze
python harness/run_all_queries.py --dataset DATASET --split SPLIT --agent codex --experiment EXPERIMENT
python harness/aggregate.py --dataset DATASET --split SPLIT --input results/EXPERIMENT/predictions --output results/EXPERIMENT/predictions.jsonl
python harness/evaluate.py --dataset DATASET --split SPLIT --pred results/EXPERIMENT/predictions.jsonl
```

- `harness/dataset.py` 统一读取 dataset.json、验证 split、筛选 videos/queries/GT；不解释名称含义。
- `task.json`、prediction、experiment.json、run_metadata、metrics 都记录 split，实验还记录 dataset 和 agent。
- 同一实验禁止混用 dataset、split、Agent 或配置。聚合核验 Query/video 成员关系和各运行的 dataset/split provenance。
- `predictions.jsonl.metadata.json` 保存聚合结果的 dataset、split、SHA256 和 annotation 哈希；评测默认强制要求 sidecar，不能通过删除 provenance 降级绕过校验。外部原始 submission 只能通过显式 unverified opt-in 进入。
- Evaluation 先验证 metadata 和 has_ground_truth，再读取选定 split 的 GT；失败/缺失 Query 仍留在该 split 分母内。无 GT split 只生成预测。
- Failed run 按 `failure_kind` 区分归属（见 13.1）：Agent 侧失败终局且计零分，Harness 侧失败中止批量运行、被评测拒绝、并在修复后自动重试；`attempts` 与 `superseded_failures` 保留审计痕迹。`metrics.json` 同时给出全 split 与 `successful_only` 两套分数。
- 默认 evaluator 由 dataset.json 指向 Adapter；官方代码优先，缺省使用 dataset-specific 实现，最后才用 generic。原有 QVHighlights 多 relevant instances mAP 语义不变。
- `default_eval_split` 仅作为评测的默认值；Ingest/Freeze/Query 要求 --split 或 config.dataset.split。Unknown split 不做别名映射。
- 旧 manifest 采用迁移方案 A：明确要求重新运行 Adapter。不自动补 split。旧 Wiki 可在保持内容不变的前提下迁入新的 videos/ 布局并校验，不自动移动用户产物。
- VLM provider、endpoint、认证、timeout 和 retry 等传输参数保留在 provenance 中，但不进入 ingest hash。并发 Ingest 使用协作式取消事件，Ctrl-C 后等待运行 worker 清理退出。
- Query Agent 仅连接临时 internal Docker network，通过无密钥 allowlist proxy 访问配置中的模型 API hostname；代理不挂载 workspace，任务结束时与网络一并删除。
- Workspace 中的 `query_id` / `video_id` / `split` 是 per-experiment HMAC 别名（见 3.3.1），Wiki 标题不含 video id，临时目录名亦用别名；预测按别名校验后翻译回真实 ID 落盘。

验收测试覆盖任意 split 名称、has_ground_truth 布尔验证、unknown split、共享视频跨 split 去重、增量 Freeze、跨 split 重复 Query ID、Query/GT 隔离、无 GT 拒绝、多 GT/预测、聚合混入其他 split/dataset 拒绝、CLI help 与默认 split。原有独立进程、只读挂载、超时清理和官方评测一致性测试继续保留。
