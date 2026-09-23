# Compile / Artifact / Query 重构

设计要求见 [完整重构规格](docs/refactor-spec.md)，旧实现计划归档在 [legacy-plan](docs/legacy-plan.md)。

实现边界：

- `vmr/core` 与 `vmr/media`：通用错误、严格 JSON、哈希、进度条、媒体操作与完整采样逻辑；不依赖 compiler 方法。
- `vmr/vlm`：caption client、共享 endpoint admission、取消与重试策略；旧导入保留同模块转发。
- `vmr/artifact`：Artifact v2（兼容读取 v1）、public/internal、独立完整性验证、只读封存、内容与构建记录双重寻址 store、WikiSet；旧产物迁移位于 `vmr/compat`。
- `vmr/compiler`：Protocol、registry、事务、批处理、五种 backend；方法拥有内容身份与公开文件声明。
- `vmr/config`：YAML/extends、profile resolution、typed Compile/Query domains、v1/v2 迁移、独立 v3 配置。
- `vmr/query`：只消费 Artifact，Experiment/QueryEngine/RunRepository、alias、workspace、批处理。
- `vmr/datasets` 与 `vmr/evaluation`：数据集与预测验证、聚合、指标和评估；旧 `harness` 导入保留转发。
- `vmr/runtime`：可替换 Runtime Protocol，复用已验证的 Docker 安全策略。
- 双向构建阶段分模块；temporal edit reducer 与 SQLite persistence 分离。
- Ablation 生成和选择 WikiSet，使用统一 Artifact QueryEngine。

兼容窗口：0.2 系列保留旧导入、旧格式算法适配器和旧 CLI，最早 0.3 独立决策移除。旧实验不隐式升级。删除 legacy code 不是本次提交的动作。

关键验收测试：

- `tests/integration/test_artifact_pipeline.py`：实际五种编译算法 + fake models、统一 QueryEngine、完整聚合评测、无 compiler/harness 最小安装的 API 与 CLI 并发查询、仅注册第三方 compiler。
- `tests/unit/test_import_boundaries.py`：依赖方向约束。
- `tests/unit/test_artifact_contract.py`：schema、public surface、完整性与路径边界。
- `tests/compatibility/test_config_domains.py`：独立配置和旧配置迁移、CLI smoke。

算法输出格式、GT/Query 隔离、HMAC aliases、Prediction validation、失败归属与 evaluator 语义保持不变。
