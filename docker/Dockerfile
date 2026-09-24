FROM node:22-bookworm-slim

# Explicit versions are required. Record the built image ID in experiment metadata.
ARG CODEX_VERSION
ARG CLAUDE_VERSION
# ffmpeg and jq serve agentic ingest, which extracts its own frames. A query
# agent cannot use them to re-extract anything: its workspace holds the frozen
# wiki only, and no video is ever mounted into it.
RUN test -n "$CODEX_VERSION" && test -n "$CLAUDE_VERSION" \
    && apt-get \
       -o Acquire::ForceIPv4=true \
       -o Acquire::Retries=3 \
       -o Acquire::http::Timeout=30 \
       -o Acquire::https::Timeout=30 \
       update \
    && apt-get \
       -o Acquire::ForceIPv4=true \
       -o Acquire::Retries=3 \
       -o Acquire::http::Timeout=30 \
       -o Acquire::https::Timeout=30 \
       install -y --no-install-recommends ca-certificates git ripgrep python3 \
       ffmpeg jq python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g "@openai/codex@${CODEX_VERSION}" "@anthropic-ai/claude-code@${CLAUDE_VERSION}" \
    && npm cache clean --force

# Keep runtime helpers inside the pinned image, not in a host-repo bind.
COPY --chmod=0755 docker/bin/ffmpeg /usr/local/bin/ffmpeg

COPY --chown=node:node docker/egress_proxy.py /opt/vmr/egress_proxy.py

USER node
WORKDIR /workspace
