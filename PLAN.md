# VMR Wiki 实现计划

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
│       └── <video_id>/
│           ├── wiki.md
│           ├── frames.jsonl
│           └── frames/
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

## 5. Dataset Adapter

不同 VMR 数据集原始 annotation 格式不同。

Harness 内部统一转换成三个 manifest：

```text
videos.jsonl
queries.jsonl
ground_truth.jsonl
```

### 5.1 videos.jsonl

```json
{"video_id":"video_001","video_path":"videos/video_001.mp4","duration":632.4}
```

### 5.2 queries.jsonl

```json
{"query_id":"q001","video_id":"video_001","query":"When does the man open the refrigerator?"}
```

`queries.jsonl` 不包含 Ground Truth 时间信息。

### 5.3 ground_truth.jsonl

一个 Query 可以包含多个 GT moments：

```json
{
  "query_id": "q001",
  "video_id": "video_001",
  "moments": [
    {"start_sec": 12.0, "end_sec": 15.5},
    {"start_sec": 48.2, "end_sec": 51.0},
    {"start_sec": 103.4, "end_sec": 106.8}
  ]
}
```

如果原数据集只有一个 Ground Truth moment，也保持统一数组形式：

```json
{
  "query_id": "q002",
  "video_id": "video_001",
  "moments": [
    {"start_sec": 73.2, "end_sec": 75.8}
  ]
}
```

这样 Harness 不需要为单 moment / 多 moment 分别设计两套 schema。

其中：

```text
queries.jsonl
```

可以进入 Query Harness。

而：

```text
ground_truth.jsonl
```

只能由 evaluator 读取。

---

## 6. Ingest 阶段

### 6.1 接口

单视频：

```bash
python harness/ingest.py \
  --video /path/to/video.mp4 \
  --video-id video_001 \
  --output wiki/<dataset>/video_001
```

批量：

```bash
python harness/ingest_all.py \
  --dataset <dataset_name>
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
wiki/<dataset>/<video_id>/
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

所有视频完成 Ingest 后：

```text
wiki/
```

进入只读状态。

正式 Query 实验开始后：

- 不重新 caption；
- 不更新 Wiki；
- 不允许 Agent 修改 Wiki；
- 同一视频的所有 Query 使用完全相同的 Wiki 版本。

建议记录每个 Wiki 的 SHA256：

```text
video_id
wiki_hash
frames_jsonl_hash
```

确保 Query 实验过程中输入没有发生变化。

---

## 9. Query Harness

对于：

```json
{
  "query_id": "q001",
  "video_id": "video_001",
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

每个 Query 只允许产生一个最终结果文件：

```text
output/prediction.json
```

但一个结果文件中可以包含多个预测 moments。

统一格式：

```json
{
  "query_id": "q001",
  "video_id": "video_001",
  "moments": [
    {
      "start_sec": 12.0,
      "end_sec": 15.0,
      "score": 0.92,
      "evidence": "The man opens the refrigerator around this interval."
    },
    {
      "start_sec": 48.0,
      "end_sec": 51.5,
      "score": 0.81,
      "evidence": "A second refrigerator-opening action occurs here."
    }
  ]
}
```

`moments` 必须按：

```text
score 从高到低
```

排序。

Evaluator 真正依赖的字段是：

```text
query_id
video_id
moments[].start_sec
moments[].end_sec
moments[].score
```

`evidence` 用于后续 error analysis，不参与标准 VMR 指标计算。

---

## 13. Prediction JSON Schema 约束

`prediction.json` 必须满足：

```text
query_id: string
video_id: string
moments: array
```

每个 prediction moment：

```text
start_sec: number
end_sec: number
score: number
evidence: string
```

并满足：

```text
start_sec >= 0
end_sec > start_sec
0 <= score <= 1
len(moments) <= max_predictions
```

同时：

```text
moments[i].score >= moments[i+1].score
```

即预测必须按 score 降序排列。

允许：

```json
"moments": []
```

用于 Agent 明确认定没有可信 moment 的情况。

如果 JSON：

- 不存在；
- 无法解析；
- 字段缺失；
- moment 数量超过限制；
- 数值非法；
- 排序非法；

则该 Query 记为 failed run。

不要自动让另一个 Agent 修复结果，否则会破坏每个 Query 一次独立运行的定义。

---

## 14. Agent Runner

统一接口：

```bash
python harness/run_all_queries.py \
  --dataset <dataset_name> \
  --agent codex \
  --experiment codex_wiki_v1
```

内部逻辑：

```text
for query in queries:

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
  --input results/codex_wiki_v1/predictions \
  --output results/codex_wiki_v1/predictions.jsonl
```

聚合后仍然保留多个 moments：

```json
{"query_id":"q001","video_id":"video_001","moments":[{"start_sec":12.0,"end_sec":15.0,"score":0.92},{"start_sec":48.0,"end_sec":51.5,"score":0.81}]}
{"query_id":"q002","video_id":"video_001","moments":[{"start_sec":73.0,"end_sec":76.0,"score":0.88}]}
```

Aggregate 阶段不能只保留 top-1，除非某个具体 evaluator 明确要求 top-1。

---

## 18. Evaluation

Evaluator 完全独立于 Agent。

执行：

```bash
python harness/evaluate.py \
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
Failed Runs
Average number of predictions per query
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
