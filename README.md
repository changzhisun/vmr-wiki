# VMR Wiki

将视频编译为与 Query 无关的不可变 Wiki Artifact，再让每条 Query 在独立 Codex / Claude Code 容器中定位视频片段。

```text
Video → Compiler → Sealed Wiki Artifact → WikiSet → Query → Prediction → Evaluate
```

支持 `simple`、`dense`、`hierarchical`、`bidirectional`、`agentic` 五种 compiler。不同方法共享 Artifact v2（兼容读取 v1） 和同一个 QueryEngine；Query 不需要 compiler 代码、compile 配置或原始视频。数据集支持 Monitor、QVHighlights 和 UCA-VMR，下文以 Monitor 为例。

## 安装

需要 Python 3.10+；Compile 内置方法需要 FFmpeg / FFprobe，真实 Agent 运行需要 Docker。运行时依赖 PyYAML。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
docker build -f docker/Dockerfile \
  --build-arg CODEX_VERSION=0.153.4 --build-arg CLAUDE_VERSION=2.1.158 \
  -t vmr-wiki-agents:local .
```

FFmpeg 的非交互包装脚本随 Agent 镜像提供，不再挂载宿主仓库中的 `docker/bin`。从旧版本升级时，请重新执行上面的镜像构建命令。

CLI 安装后使用 `vmr`，也可以使用 `python -m vmr`。模型凭据由环境变量提供，配置只保存变量名；Query 不挂载宿主机登录状态。

## 数据准备

保持 Adapter-owned contract，Adapter 输出 `dataset.json`、`videos.jsonl`、`queries.jsonl` 和有标签 split 的 `ground_truth.jsonl`。

```bash
python adapters/monitor.py \
  --annotations /path/to/dev.jsonl --split dev \
  --video-root /path/to/videos --output datasets/monitor
```

多个 split 可以用 `--annotation-dir` 一次发现全部 `<split>.jsonl`，用 `--default-eval-split` 指定评测省略 `--split` 时使用的有标签 split。

Split 名称是 opaque string，不自动映射或合并；视频须与标注时间轴一致。Query、GT 均保留多个 moments。Monitor 使用 `generic` evaluator 和 `any_acceptable_moment` 的 GT 语义。其他 Adapter 和官方评测说明见 [原有数据集文档](docs/legacy-workflow.md)。

## Compile

默认配置是 [configs/compiler/base.yaml](configs/compiler/base.yaml)。该文件只保留 dataset、路径、batch 和共享 media，通过 `extends` 组合 [configs/compiler/methods/](configs/compiler/methods/) 里的方法配置。模型、endpoint、凭据和 prompt 写在对应的 method 文件里。切换 compiler 只需改这一行：

```yaml
extends: methods/agentic.yaml  # simple | dense | hierarchical | bidirectional | agentic
```

v3 配置将 compile 和 query 分开；原 v1/v2 配置仍可显式迁移读取。Monitor 的默认 split 是 `dev`，需要别的 split 时再传 `--split`。

```bash
vmr compile --dataset datasets/monitor \
  --output-set wiki_sets/monitor-dev-agentic.json --jobs 4
```

成功构建自动校验、封存、发布，不需要单独 freeze。失败不会发布不完整 Artifact，完成的请求 checkpoint 可在重跑时复用；共享配置错误立即停止，其他错误受连续失败熔断限制。失败记录在 artifact store 的 `.compile-failures/`。

```text
artifacts/sha256-<content-id>/<manifest-hash>/
├── artifact.json       # host-only manifest
├── public/             # manifest 声明的全部 Query 输入
│   ├── wiki.md
│   └── ...
└── internal/           # 构建记录、原始回答、审计
```

Artifact ID 绑定 source、compiler 内容身份、公开表面和公开文件哈希，区分同一配置的不同模型输出；审计时间戳不改变内容 ID。不同构建记录由完整 manifest hash 区分并保留，WikiSet 和构建缓存同时锁定这两个值。构建缓存键不包含 endpoint、重试、并发等 transport 参数。WikiSet 保存 video → Artifact ID 和 manifest hash 映射；store 路径相对于 WikiSet 解析，复制时保留相对布局或显式设置 store。

```bash
vmr artifact validate artifacts/sha256-<content-id>/<manifest-hash>
vmr wikiset validate wiki_sets/monitor-dev-agentic.json
```

## Query

默认配置是 [configs/query/base.yaml](configs/query/base.yaml)，只包含 Query 所需的 profile、运行参数和路径。填写真实 Agent model/endpoint/白名单并设置凭据。

```bash
vmr query --dataset datasets/monitor \
  --wiki-set wiki_sets/monitor-dev-agentic.json \
  --experiment results/bidirectional-codex-01 --jobs 4
```

切换构建方法只需切换 WikiSet。`--query-id` 可以只执行一条查询。`query.input_mode: text` 只复制 manifest 的 `text_files`；`multimodal` 额外复制 `multimodal_files`。路径使用显式文件列表，不支持 glob。

每条 Query 使用独立 workspace 和 Agent 进程，真实 video/query/split ID 通过 HMAC 别名隐藏；只将 `public/` 的声明文件提供给 Agent。运行前、复制后和接受结果前检查完整性，保持严格 Prediction 校验与失败分类。源 Artifact 不会被修改。

实验 pin WikiSet、Artifact IDs、manifest hashes、Query 配置、模板和执行代码；构建配置不参与 Query resume。成功记录校验后复用，失败记录保留历史并重试。旧实验不能直接 resume 为新版实验，必须使用新目录。

## Evaluate

```bash
vmr evaluate --dataset datasets/monitor \
  --experiment results/bidirectional-codex-01
```

生成 `predictions.jsonl`、provenance sidecar 和 `metrics.json`。evaluator 由 Adapter 声明，Monitor 使用 `generic`；保持原来的 evaluator 输入输出和全 split 分母，基础设施错误不会静默算成模型零分。仅 QVHighlights 数据集可通过 `--official-root /path/to/moment_detr` 改用官方 evaluator，需安装 `.[official]`。

## 对照实验

```bash
python harness/ablation.py prepare --config config.yaml --dataset monitor --split dev \
  --limit-videos 10 --seed 0 --output experiments/caption_dev
python harness/ablation.py run --suite experiments/caption_dev --stage compile --jobs 4
python harness/ablation.py run --suite experiments/caption_dev --stage query --jobs 4
python harness/ablation.py run --suite experiments/caption_dev --stage evaluate
python harness/ablation.py summarize --suite experiments/caption_dev
```

每个 variant 生成独立 WikiSet，共享 content-addressed store。历史报告的 `ingested_videos` 和 `stage_wall_sec.ingest` 字段继续保留，避免静默改变消费者 schema。

## 迁移和扩展

旧 Wiki 必须显式转换，原目录保持不变：

```bash
vmr artifact migrate /old/wiki/video --artifact-store artifacts
```

通过 `vmr.artifact.wikiset.write_wikiset()` 将返回的 Artifact 按真实 video ID 关联到新 WikiSet。Host manifest 和 private files 永不进入 Agent workspace。迁移器不重新调用模型或重新解释旧配置。

旧 `harness/` Python 导入和 CLI 保留兼容窗口，其历史流程集中在 `vmr.compat`；采样、VLM 和进度条由对应的新模块实现。旧 CLI 发出 deprecation warning；新实验以 `vmr` CLI 为准。至少保留整个 0.2 系列，0.3 以后经过独立 release 决策和迁移验收再删除，当前不删除旧产物或旧实验。

新增方法实现 `WikiCompiler` 的 `parse_config / content_identity / compile` 并向 registry 注册即可。`CompileResult` 声明自身 public files，Query 和 Artifact 无需改动。直接产生符合 Artifact v1/v2 的外部系统也能使用相同 QueryEngine。协议、schema 和依赖边界见 [架构文档](docs/architecture.md)。

## 测试

```bash
python -m pytest -q                         # 全部离线测试
python -m pytest -q -m 'unit or compatibility'
python -m pytest -q -m integration
VMR_TEST_DOCKER=1 python -m pytest -q tests/test_docker_integration.py
VMR_OFFICIAL_ROOT=/path/to/moment_detr python -m pytest -q tests/test_official_parity.py
```

默认测试不调用模型 API；媒体测试使用 FFmpeg 生成短视频。覆盖五种 compiler 的同一 QueryEngine、无 compiler 安装的 Artifact portability、import boundaries、路径和 symlink 拒绝、tamper detection、resume 和 legacy config migration。Docker / 官方源码检查分别显式启用。

新包和新增测试统一使用 Ruff 0.15.16 格式化，CI 会检查格式：

```bash
ruff format vmr tests/unit tests/integration tests/compatibility
ruff format --check vmr tests/unit tests/integration tests/compatibility
```
