"""Export optional node embeddings beside, never inside, a frozen Wiki.

Writes standard float32 .npy plus a row-to-node JSONL index without requiring
NumPy/FAISS in the ingest runtime. Any NumPy/FAISS consumer can load the output.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time
import urllib.error
import urllib.request

from harness.common import (HarnessError, cli, nonempty, number, object_hash,
                            positive_int, read_jsonl, write_json, write_jsonl)
from harness.freeze import tree_hashes, verify_wiki


def node_text(node: dict) -> str:
    return "\n".join(f"{key}: " + (", ".join(node[key]) if isinstance(node[key], list) else node[key])
                     for key in ("title", "summary", "actors", "actions", "objects", "state_before", "state_after"))


class EmbeddingClient:
    def __init__(self, model, base_url, api_key_env):
        self.model = nonempty(model, "embedding model")
        self.base_url = nonempty(base_url, "embedding base_url")
        self.key = os.environ.get(api_key_env)
        if not self.key:
            raise HarnessError(f"Set {api_key_env} before exporting embeddings")

    def embed(self, texts):
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/embeddings",
            data=json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                rows = result["data"]
                if (not isinstance(rows, list) or len(rows) != len(texts) or
                        any(not isinstance(r, dict) or type(r.get("index")) is not int for r in rows) or
                        {r["index"] for r in rows} != set(range(len(texts)))):
                    raise HarnessError("Embedding response has missing or duplicate input indices")
                return [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                    raise HarnessError(f"Embedding request failed with HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise HarnessError("Embedding request failed: connection error or timeout") from None
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise HarnessError("Invalid embedding response") from exc
            time.sleep(min(2 ** attempt, 10))
        raise AssertionError("unreachable")


def npy_header(count: int, dimensions: int) -> bytes:
    header = repr({"descr": "<f4", "fortran_order": False, "shape": (count, dimensions)}).encode("ascii")
    padding = (-(10 + len(header) + 1)) % 64
    header += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header


def export_embeddings(wiki: Path, output: Path, *, model: str, base_url: str,
                      api_key_env="OPENAI_API_KEY", batch_size=32, client=None):
    positive_int(batch_size, "batch_size")
    nonempty(model, "embedding model")
    nonempty(base_url, "embedding base_url")
    wiki, output = wiki.resolve(), output.resolve()
    if output == wiki or wiki in output.parents:
        raise HarnessError("Embedding output must be outside the frozen Wiki")
    if output.exists():
        raise HarnessError("Embedding output already exists; choose a new directory")
    seal = verify_wiki(wiki)
    if "nodes.jsonl" not in seal["files"]:
        raise HarnessError("Embeddings require a hierarchical Wiki with nodes.jsonl")
    nodes = read_jsonl(wiki / "nodes.jsonl")
    if not nodes:
        raise HarnessError("Cannot embed an empty semantic tree")
    client = client if client is not None else EmbeddingClient(model, base_url, api_key_env)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".embeddings-", dir=output.parent))
    try:
        dimensions = None
        index = []
        with (staging / "vectors.npy").open("wb") as stream:
            for start in range(0, len(nodes), batch_size):
                batch = nodes[start:start + batch_size]
                texts = [node_text(node) for node in batch]
                vectors = client.embed(texts)
                if not isinstance(vectors, list) or len(vectors) != len(batch):
                    raise HarnessError("Embedding vector count differs from input count")
                for node, text, vector in zip(batch, texts, vectors):
                    if not isinstance(vector, list) or not vector:
                        raise HarnessError("Embedding vectors must be nonempty numeric lists")
                    values = [number(value, "embedding value") for value in vector]
                    norm = math.hypot(*values)
                    if not math.isfinite(norm) or norm == 0:
                        raise HarnessError("Embedding vector must have a finite nonzero norm")
                    if dimensions is None:
                        dimensions = len(values)
                        stream.write(npy_header(len(nodes), dimensions))
                    if len(values) != dimensions:
                        raise HarnessError("Embedding dimensions changed between nodes")
                    stream.write(struct.pack(f"<{dimensions}f", *(value / norm for value in values)))
                    index.append({"row": len(index), "node_id": node["node_id"], "level": node["level"],
                                  "start": node["start"], "end": node["end"], "text_sha256": object_hash(text)})
        if verify_wiki(wiki)["wiki_hash"] != seal["wiki_hash"]:
            raise HarnessError("Wiki changed during embedding export")
        write_jsonl(staging / "index.jsonl", index)
        manifest = {"version": 1, "model": model, "base_url": base_url,
                    "wiki_hash": seal["wiki_hash"], "nodes_sha256": seal["files"]["nodes.jsonl"],
                    "count": len(nodes), "dimensions": dimensions, "normalized": True,
                    "dtype": "float32", "files": tree_hashes(staging)}
        write_json(staging / "manifest.json", manifest)
        # Do not replace another writer's completed output.
        if output.exists():
            raise HarnessError("Embedding output appeared concurrently")
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = export_embeddings(args.wiki, args.output, model=args.model, base_url=args.base_url,
                               api_key_env=args.api_key_env, batch_size=args.batch_size)
    print(f"Exported {result['count']} node vectors ({result['dimensions']} dimensions) to {args.output}")


if __name__ == "__main__":
    cli(main)
