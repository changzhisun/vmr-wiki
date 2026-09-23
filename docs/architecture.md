# Architecture and versioned contracts

Compiler owns construction. Artifact owns the contract. Query owns execution. Experiment pins artifacts.

## Dependencies

`core ← media/compiler/artifact/query`; `compiler → artifact`; `query → artifact/runtime`. Neither Query nor Artifact imports compiler code. The minimal-install test physically omits compiler modules, build configuration, `vmr.compat` and `harness`. It exercises both the Experiment API and the `vmr query` module entrypoint, including v3 config loading and parallel batch execution; only the container runtime is replaced by a fixture. The test also constructs the production Docker command and checks that all bind sources exist in the stripped installation. An import blocker rejects fallback to build/legacy packages. Runtime helpers, including the noninteractive FFmpeg wrapper, ship inside the pinned image rather than a host-repository mount. CLI imports command services lazily.

`vmr.compat` retains the historical APIs for the 0.2 compatibility window. Method bodies still use local dictionary options through their backend adapter to preserve existing prompt, checkpoint and content-identity semantics; the public config and new Compile/Query services use dataclasses. These private implementation details do not cross the Artifact ABI.

## Artifact v2 (v1 remains readable)

The manifest requires exactly: `schema_version=2`, `artifact_type="vmr.wiki"`, `artifact_id`, `source`, `compiler`, `query_surface`, `content_hashes`, `provenance`, `sealed=true`.

- Source: SHA256 of video bytes and positive duration.
- Compiler: name, positive integer version, content-config SHA256.
- Query surface v1: explicit `text_files` and `multimodal_files`, relative to `public/`. Lists are disjoint, unique and exhaustive for public files. `wiki.md` is required as text, with anonymous first line `# Video`. Text mode excludes the media list.
- Content hashes cover every regular file in `public/` and `internal/`. Symlinks, special files, traversal, absolute paths and unknown structural versions/fields are rejected.
- v2 identity includes schema, source, compiler, public surface and **public** content hashes. Audit bytes and provenance do not change the Wiki content ID; they are pinned by the full manifest hash. v1 retains its original identity algorithm, including internal hashes, and is never silently rewritten. Build key separately hashes source/compiler/content options, enabling completed-build reuse without conflating a recipe with a nondeterministic result.
- Provenance is host-only; WikiSet pins the full manifest hash, including provenance. A standalone artifact verifies payload integrity; trust in a particular artifact comes from the externally pinned ID/manifest hash.
- Files and directories are sealed read-only. On macOS the top directory must briefly be writable to rename; all payload files remain sealed and readers reject that transient directory state. Publication has a complete validated payload before the final rename and seal. Publishers serialize each exact manifest with a persistent OS lock. After an interruption following rename, a retry may seal the root directory only after matching the complete manifest and verifying every payload hash and descendant permission; changed content or writable descendants are rejected.

`WikiArtifact.open`, `verify`, `verify_public`, `artifact_id`, `duration_sec`, `public_hashes`, `copy_public_to` form the consumer API. No original video or build config is required.

New builds live at `store/sha256-<content-id>/<manifest-hash>/{artifact.json,public/,internal/}`. Different audit/provenance records for the same public content remain separately addressable; publication deduplicates only an exact manifest, after verifying it. Build indexes and WikiSets select by both content ID and manifest hash. Opening by ID alone is permitted only when there is one build; ambiguous IDs require a manifest hash. Legacy v1 directories at `store/sha256-<id>/` remain readable. Each build bundle is self-contained and currently repeats its public bytes, trading disk space for portable exact build records; stable identity does not imply physical block deduplication.

Public surfaces enumerate each file explicitly rather than accepting `frames/**`. This keeps validation unambiguous but makes both the surface list and content-hash map grow linearly with frame count (about 3,600 entries each for one hour at 1 fps).

Query fully verifies a pinned build when opening it and again before accepting a result. Copying rechecks the public surface and hashes the copied bytes, and the workspace checks immutable inputs after runtime execution. Private audit bytes are read twice per successful query, down from five full-tree checks. Initialization still verifies the selected WikiSet once. No long-lived “verified” flag exempts a handle from tamper checks. `copy_public_to` validates public input only; callers needing audit verification use `verify` or `ArtifactStore.open`.

The host must ensure external Wiki public text is identity-neutral; the reader can enforce the anonymous title and filesystem boundary, but cannot infer every possible benchmark identifier embedded in arbitrary prose. Built-in agentic validation retains its identity checks.

## WikiSet v1

`schema_version`, `dataset`, opaque `split`, relative `artifact_store`, `artifacts` (video → ID), and `manifest_hashes` (video → full manifest hash). The fingerprint excludes only the store location. Query requires exact selected-split membership and verifies every pinned artifact.

Snapshots never overwrite an existing different WikiSet. To compare methods, produce another set pointing into the same store. Copy the store and set preserving relative paths; dataset query manifests are still needed, source videos are not.

## Query experiment v3

Pins dataset/selected annotations, Query configuration, WikiSet/Artifact IDs and manifest hashes, template and execution-source hashes, runtime provenance, and alias secret. The version is independent of Artifact, Query Surface, WikiSet and Prediction schemas. Unknown/legacy experiment versions cannot resume.

Artifact portability and experiment resume are distinct contracts. A copied artifact can start a new experiment in a compiler-free installation. Resuming an existing experiment additionally requires matching execution source, templates, config, runtime, dataset and pins. The source hash covers all Python files under `vmr/query`, `vmr/runtime`, `vmr/artifact`, `vmr/core`, `vmr/datasets` and `agents`; trimming those trees changes it even when the removed code is not exercised. The snapshot also compares the absolute `dataset_path`. Cross-machine resume is possible when these inputs and paths match, but an arbitrary minimal installation or relocation is not guaranteed to resume.

Creation git commit and storage paths live in `query-config.json` provenance. Compile code/configuration is excluded from execution source identity. Saved predictions have per-query SHA256, attempts and superseded failure history; only validated successful predictions are aggregated.

## Shared services and import boundaries

All frame sampling (uniform samples, source timestamps, visual-change analysis, mixed samples and context bounds) lives in `vmr/media/sampling.py`. Caption requests live in `vmr/vlm/client.py`; endpoint admission, cancellation and retry policy live in `vmr/vlm/transport.py`. Compilation provenance includes these VLM sources. Query and compile progress uses `vmr/core/progress.py`.

Compiler, media, VLM, core and Query modules import the owning `vmr` services directly, with an import-boundary test rejecting `harness` or `vmr.compat` dependencies. The old ingest driver lives in `vmr/compat/ingest.py`. Historical sampling/VLM/progress imports alias the new modules, preserving shared state and monkeypatch behavior.

## Compiler contract

`parse_config(raw) → CompilerConfig`; `content_identity(config, source) → mapping`; `compile(CompileContext) → CompileResult`. The registry is explicit, without plugin discovery. Builtins additionally resolve v3 profiles/options. A programmatic external compiler needs only the minimal three-method protocol and registration.

The service owns staging, source verification, lock, CAS publication and cache index. On Linux/macOS, a persistent `.compile.lock` inode is held with `flock` through the whole transaction. Process exit releases ownership; empty files or old PID text are not treated as owners, and lock files are never unlinked. Agentic logs live under each build key in `.work/<build-key>/logs/`. The backend owns its output files and declaration of text/media files. Private completed request checkpoints survive failures. A failed compile never creates a WikiSet claiming complete coverage.

Bidirectional stages share an explicit builder context but their implementation lives in topdown, bottomup, reconciliation, boundaries, review and render modules. Graph edits use a pure copy-on-write reducer; SQLite persistence commits only a completely validated state and its edit logs.

## Compatibility

Dataset contracts now live in `vmr/datasets`; aggregation, result provenance, metrics and evaluation live in `vmr/evaluation`. Historical `harness` imports delegate to them. Legacy identity diffs and frozen-Wiki migration live in `vmr/compat/identity.py` and `vmr/compat/artifact_migrate.py`. Generic hashing remains in `vmr/core/hashing.py`; no separate `config/fingerprint.py` is needed yet. Runtime egress policy stays with the audited Docker runtime rather than splitting into an additional `egress.py` solely to match the proposed tree.

v1/v2 public configs migrate through an isolated legacy loader; v3 compiles and queries independently. Named v2 `profiles.agents` with `query.agent_profile` and `wiki.agent_profile` remain supported alongside shared `profiles.agent` configurations, but the two forms cannot be mixed. Query-only loading of named v2 profiles requires no compiler or compatibility modules. Old imports re-export moved implementations. Old query/workspace code remains only for existing legacy callers, never as a fallback inside the new QueryEngine. Artifact migration explicitly verifies frozen legacy bytes and places only the legacy declared public surface in the new artifact.

Do not resume an old experiment with the new model. Create a new experiment after explicit artifact migration. Removal is a separate release decision after the 0.2 window, at earliest 0.3.

## Formatting

Ruff 0.15.16 formats `vmr/` and the new `tests/unit`, `tests/integration` and `tests/compatibility` directories, using the pinned Python 3.10 target and 88-column style in `pyproject.toml`. CI runs `ruff format --check` on the same scope in addition to the syntax/undefined-name checks across all Python packages and tests.
