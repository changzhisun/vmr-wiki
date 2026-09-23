VMR Wiki 重构计划

状态：Proposed
目标：将当前以 ingest.py 为中心的流程重构为 Compile → Immutable Wiki Artifact → Query → Evaluate，彻底解耦 Wiki 构建与 Query 执行，并为不同 Wiki 构建方法提供稳定、统一、可扩展的接口。

1. 背景与问题定义

当前仓库已经具备较完整的 VMR Harness 能力：Dataset Adapter、视频 Wiki 构建、Freeze/完整性校验、Query Agent 隔离执行、Prediction validation、聚合与评测等。但随着 Wiki 方法从 simple / dense 扩展到 hierarchical / bidirectional / agentic，代码逐渐形成了几个结构性问题。

最重要的问题不是单个文件过大，而是模块边界和依赖方向不够稳定。

当前存在的典型耦合包括：

harness/run_query.py 读取每个 Wiki 的 ingest.json，并通过 ingest_content_hash() 与当前完整配置比较；因此 Query 必须理解 Compile/Ingest 的配置语义。

harness/workspace.py hardcode wiki.md、nodes.jsonl、observations.jsonl、bottomup_observations.jsonl、coverage.jsonl 等不同 Wiki method 的文件名；因此新增一种 Wiki 方法可能需要修改 Query 层。

harness/common.py 中的 ingest_content_hash() 知道不同 Wiki method 的配置结构；所谓 common 实际依赖具体 method。

hierarchy.py、agentic.py、bidirectional_io.py、sampling.py 会从 harness.ingest 导入媒体工具或 JSON 辅助函数；method implementation 反向依赖顶层 orchestrator。

ingest.py 同时承担媒体处理、采样、VLM 调用、方法调度、metadata、恢复、发布等职责，成为事实上的中心模块。

freeze.py、Query、Ablation、测试等大量代码直接依赖 ingest.json / ingest_content_hash，使 Ingest 概念渗透到整个系统。

这些耦合会导致：

新增 Wiki method 时需要修改多个核心模块。

Query 无法真正独立消费不同来源的 Wiki。

Compile config 与 Query experiment identity 混在一起。

Wiki 文件格式缺少明确的稳定 ABI/contract。

很难把 Wiki artifact 拷贝到另一台机器，仅运行 Query。

内部实现继续增长后，config.py、ingest.py、BidirectionalBuilder、Experiment 会进一步成为 God Object / God Module。

本次重构的核心不是重新实现算法，而是重新定义这些边界。

2. 核心设计目标

2.1 新的顶层模型

将当前：

Dataset → Ingest → Freeze → Query → Evaluate

重构为：

Dataset / Video
      ↓
Wiki Compiler
      ↓
Immutable Wiki Artifact
      ↓
WikiSet / Snapshot
      ↓
Query Engine
      ↓
Prediction
      ↓
Evaluate

核心原则：

Compiler 生产 Artifact；Query 只消费 Artifact。

Query 不应该知道 Wiki 是由 simple、dense、hierarchical、bidirectional、agentic，还是未来的第三方系统生成的。

2.2 Compile 与 Query 必须真正解耦

以下条件作为本次重构的硬性验收标准：

一个 Wiki 成功 compile 并 seal 后，把 Artifact 复制到一个不存在 compiler method 实现、不存在 compile config 的环境，Query Engine 仍然必须能够运行。

因此 Query 层禁止：

import compiler.methods.*；

import method-specific config；

调用 compile_content_hash() 来重新解释 compile config；

根据 compiler.name 做行为分支；

hardcode nodes.jsonl、observations.jsonl 等 method-specific 文件；

要求存在原始视频才能 Query；

要求存在当初的 Compile config 才能 Query。

Query 可以读取 compiler metadata 作为 provenance，但行为不能依赖它。

2.3 ingest 统一改名为 compile

ingest 更适合描述“把外部数据摄入系统”，而当前流程实际包含：

video
→ sampling
→ VLM interpretation
→ temporal reasoning
→ graph/hierarchy construction
→ reconciliation
→ rendering
→ integrity validation
→ immutable publication

这个过程更接近一个 compiler/build system。

统一词汇：

当前名称

重构后名称

ingest

compile

ingest method

compiler / compiler backend

ingest config

compile config

ingest.json

artifact.json

ingest_content_hash

artifact identity / compile identity

frozen wiki

sealed artifact / compiled artifact

freeze

seal / publish

wiki root

artifact store

frozen dataset manifest

WikiSet / WikiSnapshot

ingest failure

compile failure

CLI 推荐：

vmr compile
vmr query
vmr evaluate

内部 Python package 使用 compiler/，避免与 Python built-in compile() 名称混淆。

3. 不变项：重构期间必须保护的实验语义

本次重构原则上不改变研究结果和实验纪律。以下内容视为稳定 contract。

3.1 Dataset contract

保持：

Adapter 输出 videos / queries / ground truth manifests；

Split 仍然是 Adapter 定义的 opaque string；

Query 与 Ground Truth 分离；

一个 Query 支持多个 GT moments；

Prediction 支持多个 moments。

3.2 Query isolation

保持：

每个 Query 独立 workspace；

每个 Query 独立 Agent 进程/容器；

不读取其他 Query/Prediction/GT；

alias/HMAC 机制保持；

immutable input tamper detection 保持；

runtime security policy 保持。

3.3 Prediction contract

保持现有严格 validation 语义，包括：

JSON 格式；

duplicate key；

NaN/Infinity；

moment 区间；

score；

输出文件数量和路径约束；

failure classification。

3.4 Reproducibility

必须继续记录：

source/video identity；

artifact identity；

query experiment identity；

templates hash；

runtime provenance；

source code hash/git commit；

prediction hash。

但需要重新区分：

Artifact Identity
    决定“这是哪个 Wiki 构建结果”

Query Experiment Identity
    决定“这是在哪个 Artifact 集合上运行的哪次 Query 实验”

Provenance
    记录“它是如何、在哪里被产生/运行的”

4. 目标目录结构

推荐最终结构：

vmr-wiki/
├── vmr/
│   ├── core/
│   │   ├── errors.py
│   │   ├── jsonio.py
│   │   ├── hashing.py
│   │   └── time.py
│   │
│   ├── config/
│   │   ├── models.py
│   │   ├── loader.py
│   │   ├── migrate.py
│   │   ├── resolve.py
│   │   └── fingerprint.py
│   │
│   ├── media/
│   │   ├── ffmpeg.py
│   │   ├── probe.py
│   │   ├── frames.py
│   │   └── sampling.py
│   │
│   ├── artifact/
│   │   ├── manifest.py
│   │   ├── artifact.py
│   │   ├── integrity.py
│   │   ├── store.py
│   │   └── wikiset.py
│   │
│   ├── compiler/
│   │   ├── protocol.py
│   │   ├── registry.py
│   │   ├── context.py
│   │   ├── pipeline.py
│   │   └── methods/
│   │       ├── simple/
│   │       ├── dense/
│   │       ├── hierarchical/
│   │       ├── bidirectional/
│   │       └── agentic/
│   │
│   ├── query/
│   │   ├── experiment.py
│   │   ├── engine.py
│   │   ├── workspace.py
│   │   ├── prediction.py
│   │   ├── aliases.py
│   │   └── repository.py
│   │
│   ├── runtime/
│   │   ├── protocol.py
│   │   ├── docker.py
│   │   ├── egress.py
│   │   └── trace.py
│   │
│   ├── datasets/
│   ├── evaluation/
│   └── cli/
│
├── adapters/
├── templates/
├── tests/
└── pyproject.toml

迁移过程中不需要一次完成目录搬迁。优先保证依赖方向正确，然后再移动文件。

5. 依赖方向

最终必须满足：

core
 ↑
config        media
 ↑             ↑
 └──── compiler ─────→ artifact ←──── query
          ↑                         ↑
       datasets                  runtime
                                    ↑
                                evaluation

更具体地说：

compiler.methods.*  ───→ compiler.protocol
compiler.*          ───→ artifact
query.*             ───→ artifact
query.*             ───→ runtime

compiler  ─X─→ query
query     ─X─→ compiler
artifact  ─X─→ compiler.methods
core      ─X─→ method-specific modules

建议增加 import-boundary 测试，防止未来重新出现反向依赖。

6. Wiki Artifact：新的 Compile / Query ABI

这是本次重构最关键的抽象。

6.1 Artifact 目录

推荐：

artifacts/<artifact_id>/
├── artifact.json
├── public/
│   ├── wiki.md
│   ├── ...
│   └── frames/
└── internal/
    ├── ...
    └── audit/

含义：

artifact.json：host-side manifest，不直接提供给 Agent；

public/：Query Workspace 可以公开的全部内容；

internal/：compile checkpoints、原始模型回答、调试数据、审计等；Query 永远不可见。

这会替代当前 workspace.py 中 hardcode public filename 的机制。

6.2 Query 只复制 public/

Query workspace 不再知道 method-specific filename：

artifact = WikiArtifact.open(path)
artifact.verify()
artifact.copy_public_to(workspace / "wiki", mode=input_mode)

原则：

是否对 Query 可见，由 Artifact 自己的 public surface 决定，而不是 Query Engine 猜测文件含义。

6.3 text / multimodal 模式

需要保留当前 query.input_mode 能力。

建议 Artifact manifest 显式声明公开资源类型：

{
  "query_surface": {
    "version": 1,
    "text": [
      "wiki.md",
      "nodes.jsonl"
    ],
    "multimodal": [
      "frames.jsonl",
      "frames/**"
    ]
  }
}

或者在目录上进一步结构化：

public/
├── text/
└── media/

第一版推荐 manifest 声明方式，迁移成本更低。

7. artifact.json Schema

建议 v1：

{
  "schema_version": 1,
  "artifact_type": "vmr.wiki",
  "artifact_id": "sha256:...",

  "source": {
    "video_sha256": "...",
    "duration_sec": 126.42
  },

  "compiler": {
    "name": "bidirectional",
    "version": 1,
    "content_config_hash": "sha256:..."
  },

  "query_surface": {
    "version": 1,
    "text_files": [
      "wiki.md",
      "nodes.jsonl",
      "observations.jsonl"
    ],
    "multimodal_files": [
      "frames.jsonl",
      "frames/**"
    ]
  },

  "content_hashes": {
    "public/wiki.md": "...",
    "public/nodes.jsonl": "..."
  },

  "provenance": {
    "created_at": "...",
    "source_commit": "..."
  }
}

7.1 Host-only metadata

artifact.json 可以记录：

compiler name；

compiler version；

content-affecting config hash；

source hash；

build provenance；

integrity hashes。

但是 Agent 默认不应该看到：

原始 video_id；

dataset name；

split；

benchmark identifier；

compile config；

host path。

如果 artifact store 需要将真实 video identity 与 artifact 关联，应通过外部 WikiSet / registry 保存，而不是把 benchmark key 暴露到 public/。

7.2 Artifact ID

建议：

artifact_id = hash(
    artifact_schema_version,
    source_content_identity,
    compiler_name,
    compiler_version,
    content_affecting_compile_config,
    compiler_processing_rule_version
)

注意：

API endpoint、timeout、retry、并发等 transport 配置不应进入 content identity，除非确实会改变内容语义；

provenance 可以完整记录这些运行信息；

method-specific 内容身份由 compiler 自己负责声明，不能重新集中到 core/common.py。

8. WikiArtifact API

建议提供一个稳定的 domain object：

@dataclass(frozen=True, slots=True)
class WikiArtifact:
    root: Path
    manifest: ArtifactManifest

    @classmethod
    def open(cls, root: Path) -> "WikiArtifact": ...

    def verify(self) -> None: ...
    def artifact_id(self) -> str: ...
    def duration_sec(self) -> float: ...
    def copy_public_to(self, target: Path, *, input_mode: str) -> None: ...
    def public_hashes(self, *, input_mode: str) -> dict[str, str]: ...

Query 只依赖上述 API。

verify() 负责：

schema validation；

path traversal 防护；

symlink policy；

declared file 与实际文件一致；

hash 验证；

public/internal 边界；

immutable seal 状态。

9. WikiSet / WikiSnapshot

当前一个 dataset/video 默认映射到一个 Wiki root，不适合同时比较多种 Compile method。

引入 WikiSet：

wiki_sets/
├── qvhighlights-val-simple.json
├── qvhighlights-val-hierarchical.json
└── qvhighlights-val-bidirectional.json

示例：

{
  "schema_version": 1,
  "dataset": "qvhighlights",
  "split": "val",
  "artifacts": {
    "video_001": "sha256:aaa",
    "video_002": "sha256:bbb",
    "video_003": "sha256:ccc"
  }
}

真正存储：

artifact_store/
├── sha256-aaa/
├── sha256-bbb/
└── sha256-ccc/

Query CLI：

vmr query \
  --dataset datasets/qvhighlights \
  --split val \
  --wiki-set wiki_sets/qvhighlights-val-bidirectional.json \
  --experiment bidirectional-codex-01

切换 Wiki method 时，只切换 WikiSet，不切换 Query implementation。

9.1 Experiment pin 的是 Artifact，而不是 Compile config

Experiment manifest 应保存：

{
  "wiki_set_hash": "...",
  "artifacts": {
    "video_001": "sha256:aaa",
    "video_002": "sha256:bbb"
  },
  "query_config_hash": "...",
  "templates_hash": "...",
  "runtime": {}
}

Resume 时验证：

当前 artifact ID/hash == experiment pin 的 artifact ID/hash

不再验证：

artifact compile config == 当前 config 里的 compile config

10. Compiler Protocol

10.1 最小接口

不要引入复杂 plugin framework。使用明确的 Protocol + registry 即可。

class WikiCompiler(Protocol):
    name: str
    version: int

    def parse_config(self, raw: Mapping[str, Any]) -> CompilerConfig: ...

    def content_identity(
        self,
        config: CompilerConfig,
        source: VideoSource,
    ) -> Mapping[str, Any]: ...

    def compile(
        self,
        context: CompileContext,
    ) -> CompileResult: ...

Registry：

COMPILERS = {
    "simple": SimpleCompiler(),
    "dense": DenseCompiler(),
    "hierarchical": HierarchicalCompiler(),
    "bidirectional": BidirectionalCompiler(),
    "agentic": AgenticCompiler(),
}

顶层 compile orchestration：

compiler = registry.get(config.wiki.method)
method_config = compiler.parse_config(config.wiki.method_config)
result = compile_service.compile(video, compiler, method_config)

核心层不再包含：

if method == "simple": ...
elif method == "dense": ...
elif method == "hierarchical": ...

10.2 Compiler owns method-specific identity

当前 common.ingest_content_hash() 对不同方法进行分支。重构后：

compiler.content_identity(config, source)

负责 method-specific 内容身份。

公共代码只做：

object_hash(identity_payload)

这样新增 compiler 不需要修改 common。

11. Compile Transaction

当前用户心智是：

ingest
→ freeze

重构后改为一个原子事务：

compile staging
      ↓
method build
      ↓
validate
      ↓
generate manifest
      ↓
seal
      ↓
atomic publish
      ↓
immutable artifact

建议 API：

artifact = compile_service.compile(video, compiler, config)

内部：

.tmp/<build-id>/
   ↓
compiler 写入 public/internal
   ↓
ArtifactValidator
   ↓
计算 hashes + artifact_id
   ↓
chmod/read-only seal
   ↓
atomic rename
   ↓
artifact_store/<artifact_id>/

原 freeze.py 的核心完整性逻辑保留，但作为 artifact publish/seal phase，而不是要求用户单独运行的顶层阶段。

12. Config 重构

当前 harness/config.py 同时承担：

YAML loading；

extends；

v1/v2 compatibility；

schema validation；

profile resolution；

method config normalization；

internal legacy dict 生成；

config explain；

Wiki content hash 输出。

建议拆分。

12.1 Typed config

优先使用标准库 dataclass，不急于引入大型 validation framework。

@dataclass(frozen=True, slots=True)
class MediaConfig:
    sample_interval_sec: float
    image_max_size: int
    jpeg_quality: int

@dataclass(frozen=True, slots=True)
class CompileConfig:
    compiler: str
    media: MediaConfig
    method: CompilerConfig

@dataclass(frozen=True, slots=True)
class QueryConfig:
    agent_profile: str
    input_mode: str
    timeout_sec: float

@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig
    storage: StorageConfig
    compile: CompileConfig
    query: QueryConfig

12.2 Config pipeline

明确为：

read YAML
   ↓
resolve extends
   ↓
migrate legacy schema
   ↓
validate public schema
   ↓
resolve profiles
   ↓
construct typed config

不要继续把 v2 public config normalize 成旧 internal dict schema 作为长期形态。

12.3 Compile 与 Query config 拆开

最终允许：

vmr compile --config compile.yaml
vmr query --config query.yaml --wiki-set ...

Query config 不应包含完整 compile config。

为了迁移兼容，可以先继续支持一个完整 config.yaml，内部立即拆成：

config.compile
config.query

然后逐步允许独立配置文件。

13. common.py 拆分

当前 common.py 混合了 generic utility 与 compile-specific identity。

建议拆为：

core/errors.py
    HarnessError
    RunFailure

core/jsonio.py
    read_json
    write_json
    read_jsonl
    parse_json
    atomic_text

core/hashing.py
    file_hash
    tree_hashes
    object_hash

core/time.py
    now

ingest_content_hash() 不进入 core。

其 replacement 应位于：

compiler identity
artifact identity
config fingerprint

三个明确概念中。

14. Media 层抽离

当前 method implementation 会从 harness.ingest 反向导入：

media_command

extract_frame

unwrap_markdown_json_fence

这是需要最早处理的依赖问题之一。

目标：

media/ffmpeg.py
    media_command

media/frames.py
    extract_frame
    encode_frame

core/jsonio.py / vlm/structured.py
    unwrap_markdown_json_fence

之后：

compiler.methods.* → media

而不是：

compiler.methods.* → top-level ingest orchestrator

15. Query Engine 重构

当前 Experiment 负责太多内容：

dataset/split；

wiki readiness；

compile config compatibility；

freeze hash；

source hash；

experiment manifest；

lock；

alias；

resume；

workspace；

Agent runtime；

prediction；

run metadata；

logs。

建议拆成以下组件。

15.1 ExperimentManifest

职责：

描述不可变 experiment inputs；

pin WikiSet/artifacts；

pin Query config；

pin template/source/runtime provenance；

resume equality check。

15.2 QueryEngine

只负责编排一次 Query：

class QueryEngine:
    def run(self, query: Query) -> QueryRunResult:
        artifact = self.artifacts.for_video(query.video_id)
        artifact.verify()

        with self.workspace.create(query, artifact) as workspace:
            execution = self.runtime.execute(workspace)
            prediction = self.predictions.read(execution)

        return self.runs.complete(query, execution, prediction)

15.3 WorkspaceFactory

只负责：

opaque query/video aliases；

task.json；

templates；

artifact public/ copy；

readonly permissions；

tamper checks；

cleanup。

它不应知道 nodes.jsonl、coverage.jsonl 等名字。

15.4 RunRepository

负责：

run metadata；

attempts；

interrupted recovery；

superseded failures；

prediction hash；

logs paths。

15.5 PredictionReader

只负责：

output directory contract；

parsing；

validation；

alias reverse mapping。

16. Runtime 重构

agents/runner.py 可拆为：

runtime/protocol.py
runtime/docker.py
runtime/egress.py
runtime/trace.py

定义：

class AgentRuntime(Protocol):
    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def execute(self, workspace: Path, ...) -> ExecutionResult: ...

测试中可以直接使用 FakeRuntime。

注意：Docker security policy 必须保持集中可审计，不要为了抽象把以下规则分散到多个 builder：

read-only filesystem；

capability drop；

no-new-privileges；

network isolation；

egress allowlist；

credential injection；

workspace mount policy。

17. Bidirectional Pipeline 拆分

BidirectionalBuilder 当前承担多个明显不同的阶段。建议重构为 pipeline stages，而不是仅机械拆文件。

目标：

class BidirectionalPipeline:
    def compile(self, ctx):
        topdown = self.topdown.run(ctx)
        bottomup = self.bottomup.run(ctx, topdown)
        reconciled = self.reconciler.run(ctx, topdown, bottomup)
        refined = self.boundaries.run(ctx, reconciled)
        reviewed = self.reviewer.run(ctx, refined)
        return self.renderer.render(ctx, reviewed)

建议模块：

compiler/methods/bidirectional/
├── compiler.py
├── config.py
├── topdown.py
├── bottomup.py
├── reconcile.py
├── boundaries.py
├── review.py
├── model.py
└── render.py

好处：

每个阶段可独立单测；

ablation 可以组合 stage；

checkpoint 更容易定义；

减少超大 class state；

method-specific I/O 不污染公共 compile service。

18. Temporal Graph 重构

TemporalGraph.apply() 建议拆成：

EditCommand
   ↓
validate_edit(state, command)
   ↓
apply_edit(state, command)
   ↓
GraphState
   ↓
GraphRepository.persist(state)

目标是把图编辑尽量变为 pure function。

这样 graph correctness 测试不需要 SQLite/文件系统，能显著提高测试速度和可推理性。

19. Simple / Dense / Hierarchical / Agentic 迁移原则

每个方法迁移后必须满足同样的 compiler contract，但允许拥有不同 public files。

例如：

simple artifact public/
    wiki.md
    frames.jsonl
    frames/

hierarchical artifact public/
    wiki.md
    nodes.jsonl
    observations.jsonl
    frames.jsonl
    frames/

bidirectional artifact public/
    wiki.md
    nodes.jsonl
    observations.jsonl
    bottomup_observations.jsonl
    coverage.jsonl
    frames.jsonl
    frames/

Query 不需要知道这些差异。

迁移顺序建议：

simple
→ dense
→ hierarchical
→ bidirectional
→ agentic

从简单方法开始验证新的 Artifact/Compiler API，再迁移复杂实现。

20. 第三方 / 外部 Wiki 支持

目标架构应允许未来存在：

Your Compiler
Other Compiler
Human-authored Wiki
External Baseline
        ↓
Wiki Artifact v1
        ↓
Common Query Engine
        ↓
Common Evaluation

这要求 Artifact ABI 是真正稳定的边界，而不是“当前几个内部 compiler 恰好都能用”。

可以提供：

vmr artifact validate /path/to/artifact
vmr wikiset validate /path/to/set.json

外部系统只需要产生合法 Artifact，无需 import 本仓库 compiler code。

21. CLI 目标

最终推荐：

# Dataset preparation 保持 adapter-owned
python adapters/qvhighlights.py ...

# Compile
vmr compile \
  --config compile.yaml \
  --dataset datasets/qvhighlights \
  --split val \
  --output-set wiki_sets/qvhighlights-val-bidirectional.json

# Validate artifacts
vmr artifact validate artifact_store/sha256-...
vmr wikiset validate wiki_sets/qvhighlights-val-bidirectional.json

# Query
vmr query \
  --config query.yaml \
  --dataset datasets/qvhighlights \
  --split val \
  --wiki-set wiki_sets/qvhighlights-val-bidirectional.json \
  --experiment experiments/bidir-codex-01

# Evaluate
vmr evaluate \
  --dataset datasets/qvhighlights \
  --split val \
  --experiment experiments/bidir-codex-01

迁移期保留旧入口：

harness/ingest.py
harness/ingest_all.py
harness/run_query.py
harness/run_all_queries.py

但内部转发到新 service，并输出 deprecation warning。

22. 测试策略

22.1 测试层级

建议：

tests/
├── unit/
├── integration/
├── docker/
└── compatibility/

Unit

覆盖：

Artifact manifest；

identity；

config migration；

compiler stage；

graph edits；

validation；

aliases；

prediction parsing。

不需要 Docker、真实 ffmpeg、真实 API。

Integration

覆盖：

compile staging → sealed artifact；

WikiSet；

Query workspace；

resume；

tamper detection；

adapter + compile + query fake runtime。

Docker

覆盖：

filesystem isolation；

credential policy；

egress proxy；

timeout/cancel；

readonly behavior。

Compatibility

覆盖：

legacy config → typed config；

legacy ingest output → artifact migration（如果决定支持）；

old CLI wrappers；

artifact v1 stability。

22.2 最关键的架构测试

必须增加：

def test_query_has_no_compiler_dependency():
    ...

可以通过 import graph/static check 保证：

vmr.query

不 import：

vmr.compiler
vmr.compiler.methods.*

22.3 Artifact portability test

核心验收：

用 compiler 生成 Artifact；

将 Artifact + WikiSet copy 到临时目录；

从 sys.path/运行环境中移除 compiler method package；

只加载 query/runtime/artifact；

Fake Agent Runtime 完成 prediction；

Query 成功。

这条测试直接证明 Compile/Query 解耦不是名义上的。

22.4 Cross-method query test

参数化：

@pytest.mark.parametrize("compiler", [
    "simple",
    "dense",
    "hierarchical",
    "bidirectional",
    "agentic",
])
def test_every_artifact_can_use_same_query_engine(compiler):
    ...

QueryEngine implementation 在整个测试中完全相同。

23. CI 分层

当前测试中 Docker lifecycle、timeout、cancel、媒体集成天然较慢。重构后建议：

CI Fast
    unit + compatibility

CI Integration
    artifact + compile + query fake runtime

CI Docker
    docker/security/egress/timeout

CI Full
    nightly / merge gate

目标：日常改纯 Python/domain 代码时，不必等待 Docker/timeout 类慢测。

24. 迁移阶段

不建议一次“大爆炸式”重写。采用逐步替换，每一步保持主分支可运行。

Phase 0 — Characterization / Freeze Behavior

目标：先锁住现有行为。

任务：

补充关键 golden/characterization tests；

记录当前 config normalization 行为；

记录当前 artifact/wiki 文件语义；

记录 Query failure classification；

记录 freeze/tamper 行为；

建立主要 CLI smoke tests。

完成标准：

后续 PR 可以明确判断是“重构”还是“行为变化”。

Phase 1 — Core / Media Dependency Cleanup

目标：解决最明显的反向依赖，不改变公开行为。

任务：

拆 common.py；

将 media_command、extract_frame 等移入 media/；

structured JSON helper 移出 ingest.py；

hierarchy.py、agentic.py、bidirectional_io.py、sampling.py 不再 import harness.ingest；

保留兼容 re-export，避免一次改完全部调用点。

完成标准：

compiler/method implementation 不再依赖 ingest orchestrator

风险：低。

Phase 2 — Artifact v1

目标：建立稳定 Compile/Query ABI。

任务：

新增 ArtifactManifest；

新增 WikiArtifact；

引入 public/ / internal/；

把 Freeze 校验迁移为 Artifact integrity/seal；

支持从当前 Wiki 输出构造 artifact；

增加 artifact validate；

Artifact path/hash/path traversal/symlink 测试。

迁移期可以同时写：

artifact.json

和旧：

ingest.json
frozen.json

直到 Query 全部迁移。

完成标准：

一个独立 WikiArtifact 可以 verify；

不依赖 Query 或 compiler implementation。

风险：中。

Phase 3 — Query consumes Artifact only

目标：真正解除 Query → Ingest/Compiler 依赖。

任务：

Experiment 不再读 ingest.json；

删除 Query 中 ingest_content_hash() 比较；

workspace.py 删除 hardcoded public file list；

Query 通过 WikiArtifact.copy_public_to() 构造 workspace；

Experiment pin artifact ID/hash；

Query 只验证 artifact integrity + WikiSet membership；

新增 portability test；

新增 import-boundary test。

完成标准：

Query 环境删除 compiler method code 和 compile config 后仍可运行。

这是整个重构的第一核心里程碑。

风险：高，但收益最高。

Phase 4 — WikiSet / Artifact Store

目标：同一 Dataset/Split 可自由切换不同 Wiki 构建方法。

任务：

content-addressed artifact store；

WikiSet schema；

WikiSet validation；

compile 输出 WikiSet；

Query 接收 WikiSet；

ablation 改为选择 WikiSet，而不是假设某种 wiki-root layout。

完成标准：

同一 video 可同时存在 simple/hierarchical/bidirectional artifacts

并且：

切换 Wiki method 不需要修改 Query config/implementation

风险：中。

Phase 5 — Compiler Protocol / Registry

目标：新增 Compile method 不修改核心编排。

任务：

WikiCompiler Protocol；

compiler registry；

method-specific parse_config()；

method-specific content_identity()；

generic compile transaction；

generic Artifact publication。

迁移顺序：

simple → dense → hierarchical → bidirectional → agentic

完成标准：

新增一个 test compiler 只需要：

实现 protocol
注册 compiler

不需要修改：

artifact
query
workspace
common/core

风险：中高。

Phase 6 — ingest → compile

目标：完成公共概念迁移。

任务：

CLI 新增 vmr compile；

新命名 CompileContext / CompileResult / CompileFailure；

ingest.json 正式迁移为 artifact.json；

.ingest-failures → .compile-failures；

README / PLAN / examples 更新；

旧 CLI 保留 compatibility wrapper；

deprecation policy 明确。

完成标准：

README 主流程不再使用 ingest 作为 Wiki 构建概念。

风险：中，主要是命名和兼容性。

Phase 7 — Typed Config

目标：拆除 config.py 的历史内部 schema。

任务：

dataclass config；

loader / extends / migrate / resolve 分离；

compile/query config domain 分离；

compiler 自己负责 method config validation；

删除中央 method-specific normalization；

保留 legacy config migration。

完成标准：

业务代码不再大量出现：

cfg["ingest"][...]
cfg["wiki"]["method_config"][...]

而使用 typed config。

风险：高，建议在 Artifact/Query 解耦之后再做。

Phase 8 — Bidirectional / Experiment Internal Refactor

目标：在边界稳定后整理内部复杂度。

任务：

拆 Bidirectional pipeline stages；

拆 TemporalGraph reducer/persistence；

拆 Experiment / QueryEngine / RunRepository；

Runtime protocol；

slow tests 分类。

完成标准：

每个 stage 可独立单测；

Query orchestration 清晰；

Docker runtime 可替换 FakeRuntime；

domain tests 不依赖容器。

风险：中。

Phase 9 — Legacy Removal

只有在新架构稳定后进行。

候选删除：

harness/ingest.py compatibility exports；

harness/ingest_all.py；

ingest.json reader；

frozen.json legacy path；

ingest_content_hash()；

legacy normalized config schema；

old wiki-root layout；

deprecated CLI flags。

必须先明确 release/deprecation window，不在早期 PR 中删除。

25. 推荐 PR 划分

为了 code review 可控，建议按以下 PR，而不是一个超大分支。

PR 1 — Extract core/media utilities

common.py 拆分；

ingest.py 的 media/json helper 抽出；

删除 method → ingest imports；

行为不变。

PR 2 — Introduce WikiArtifact v1

manifest；

public/internal；

integrity；

seal；

compatibility adapter。

PR 3 — Query reads Artifact only

删除 Query 对 ingest config/hash 的依赖；

workspace 只看 artifact public surface；

portability tests。

PR 4 — WikiSet + Artifact Store

多方法并存；

experiment pin artifacts；

ablation 迁移。

PR 5 — Compiler Protocol + simple/dense

验证 protocol 设计。

PR 6 — hierarchical compiler migration

PR 7 — bidirectional compiler migration

PR 8 — agentic compiler migration

PR 9 — Compile CLI + terminology migration

PR 10 — Typed config / legacy migration

PR 11 — Query/Runtime internal decomposition

PR 12 — Remove deprecated ingest architecture

PR 数量可以合并，但依赖顺序建议保持。

26. 每个 PR 的通用验收规则

每个重构 PR 必须满足：

不静默改变 evaluator 输入/输出。

不降低 Query isolation。

不降低 integrity/tamper 校验。

不把 GT 引入 Compile 或 Query workspace。

不把真实 benchmark ID 暴露给 Agent。

不通过 catch-all fallback 吞掉 Harness/infra error。

旧实验不能在输入已变化时被错误 resume。

运行产物必须有明确 schema/version。

新增 method-specific 逻辑不能进入 core 或 Query。

新增公共 schema 必须有 migration/version strategy。

27. Compatibility Strategy

27.1 旧 Config

支持：

legacy config
   ↓
migrate
   ↓
new typed config

不要让新业务代码继续理解 legacy schema。

27.2 旧 Wiki

有两个选择。

推荐方案：显式迁移

提供：

vmr artifact migrate /old/wiki/path

产生新的 Artifact v1。

好处：

Query implementation 永远只理解 Artifact；

legacy logic 集中；

新代码干净。

不推荐方案：Query 直接兼容旧 Wiki

这会再次把旧 ingest.json/frozen.json 语义引入 Query，因此只适合非常短的过渡期。

27.3 旧 Experiment

不建议直接 resume 到新 Artifact model。

如果旧 experiment manifest 没有 artifact identity，应明确要求创建新 experiment，而不是隐式升级。

28. Schema Versioning

至少分别 version：

Artifact Manifest Schema
Query Surface Schema
WikiSet Schema
Experiment Manifest Schema
Prediction Schema（若未来变化）

不要用一个全局 version 绑定所有格式。

读者原则：

reader understands version N
→ 可以读取 N
→ 对未知更高 version 明确失败

禁止 silently ignore unknown structural fields when they change semantics。

29. Observability / Provenance

Compile 与 Query 分离后，provenance 更需要明确。

Compile provenance

记录：

compiler name/version；

source commit；

source video hash；

compile content config hash；

model/profile provenance；

runtime/endpoint metadata；

usage/timing；

audit/checkpoint info。

Query provenance

记录：

artifact IDs；

WikiSet hash；

query config；

agent image/model/runtime；

templates hash；

source code hash；

alias secret；

per-query attempts/failure history。

不要要求 Query 重新计算 Compile provenance 才允许运行。

30. Security / Isolation Review

Artifact 化后要重新检查：

Public surface

只有 public/ 内容允许进入 workspace。

Host-only metadata

真实 video_id、dataset、split 等不得因为 artifact.json 被意外复制到容器。

Paths

manifest 中的公开路径必须：

relative；

不含 ..；

不逃出 artifact root；

symlink policy 明确。

Immutable verification

至少：

before workspace copy: artifact.verify()
after workspace copy: copied hashes match declared public hashes
after agent run: immutable workspace inputs unchanged
before accepting result: source artifact still verifies

保留当前已有的 tamper failure classification。

31. Performance Considerations

本次重构主要解决架构，不应顺手大幅改变算法行为。

但 Artifact Store 可以顺便提供：

content-addressed deduplication；

completed compile reuse；

WikiSet cheap snapshot；

Artifact copy/link strategy；

public subset hash caching。

避免早期引入：

分布式 build scheduler；

remote artifact registry；

elaborate CAS protocol；

plugin discovery framework。

先把本地边界做正确。

32. 明确不做的事情

本轮重构不以以下内容为目标：

改进 VMR 算法指标；

更换 VLM/Agent；

引入复杂 dependency injection container；

将所有 function class 化；

同时重写 Dataset Adapter；

同时重写官方 evaluator；

大规模优化 FFmpeg 性能；

引入远程数据库或服务化架构；

一次性删除所有 v1 config compatibility。

原则：

先建立稳定边界，再优化内部实现。

33. Definition of Done

本次整体重构完成时，必须满足以下全部条件。

Architecture

query 不 import compiler。

compiler 不 import query。

method-specific code 不进入 core。

method implementation 不 import 顶层 compile orchestrator。

新 Wiki method 不要求修改 Query。

Artifact

每个成功 compile 都产生 versioned artifact.json。

Artifact 有明确 public/ 与 internal/ 边界。

Artifact 可以独立 verify。

Artifact 可以复制到另一环境使用。

Query 不需要 compile config。

Query

Query 只通过 Artifact API 获得 Wiki 输入。

Workspace 不 hardcode method-specific filename。

Experiment pin artifact ID/hash，而不是 compile config。

Same QueryEngine 可以消费所有 compiler 的 Artifact。

compiler package 缺失时 Query 仍能运行。

Multi-method

同一个 Dataset/Split 可以同时存在多个 WikiSet。

切换 Wiki method 只需要切换 WikiSet。

simple/dense/hierarchical/bidirectional/agentic 都实现同一 Compiler Protocol。

Terminology

README 主流程使用 Compile，而不是 Ingest。

CLI 有 vmr compile。

ingest.json 已退出新架构主路径。

ingest_content_hash() 已退出 Query 路径。

Reproducibility & Security

Artifact identity 与 Query experiment identity 明确分离。

Query isolation 不降低。

benchmark identifier aliasing 不降低。

tamper detection 不降低。

provenance 信息完整。

Tests

Unit / Integration / Docker 测试分层。

Artifact portability test 存在。

Query/compiler import-boundary test 存在。

Cross-method common QueryEngine test 存在。

legacy config migration tests 存在。

现有 evaluator parity tests 保持通过。

34. 最优先实施顺序

如果只考虑架构收益，优先级如下：

1. 抽离 core/media，消灭 method → ingest.py 反向依赖

2. 建立 WikiArtifact v1 + public/internal contract

3. Query 改为只消费 Artifact
   - 删除 ingest_content_hash comparison
   - 删除 hardcoded Wiki public filenames

4. 建立 WikiSet / Artifact Store
   - 允许不同 Compile method 独立 Query

5. 建立 WikiCompiler Protocol
   - 逐个迁移 method

6. 正式把 Ingest terminology / CLI 改为 Compile

7. Typed config

8. 拆 Bidirectional / Experiment / Runtime 内部复杂度

9. 删除 legacy compatibility

最关键的两个里程碑是：

Milestone A — Query Artifact Independence

Compile
  ↓
Artifact

      [断开 compiler code/config]

Artifact
  ↓
Query

这个测试通过后，Compile/Query 才算真正解耦。

Milestone B — Method Independence

Simple Compiler ─────────┐
Dense Compiler ──────────┤
Hierarchical Compiler ───┤
Bidirectional Compiler ──┼──→ Artifact v1 ──→ Same Query Engine
Agentic Compiler ────────┤
External Compiler ───────┘

这个测试通过后，整个 Harness 的长期扩展边界才算稳定。

35. 最终设计原则

整个重构最终应围绕下面四句话检查：

Compiler owns construction.
每个 compiler 自己负责方法特定的配置、内容身份和构建过程。

Artifact owns the contract.
Compile 与 Query 只通过 versioned immutable artifact 通信。

Query owns execution, not construction.
Query 不知道 Wiki 是怎么生成的，也不应该验证当初的构建配置。

Experiment pins artifacts, not pipelines.
实验复现依赖的是精确 Artifact identity，而不是重新解释一遍 Compile pipeline。

如果一项未来改动违反其中任何一句，应优先重新检查模块边界，而不是继续增加兼容分支。