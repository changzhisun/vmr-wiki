import ast
import io
import json
import struct

import pytest

from harness.common import HarnessError, read_jsonl, write_json, write_jsonl
from harness.embed_wiki import EmbeddingClient, export_embeddings
from harness.freeze import freeze_wiki, remove_tree, tree_hashes, verify_wiki
from harness.hierarchy import SCHEMA


@pytest.fixture
def semantic_wiki(tmp_path):
    root = tmp_path / "wiki"
    nodes = [{**SCHEMA, "node_id": f"chapter_{index}", "parent_id": None, "level": "chapter",
              "start": float(index), "end": float(index + 1)} for index in range(3)]
    write_jsonl(root / "nodes.jsonl", nodes)
    write_jsonl(root / "frames.jsonl", [])
    (root / "wiki.md").write_text("# Video\n")
    (root / "frames").mkdir()
    write_json(root / "ingest.json", {"video_id": "fixture", "duration": 3,
                                     "ingest_config_hash": "fixture", "content_hashes": tree_hashes(root)})
    freeze_wiki(root)
    yield root
    remove_tree(root)


class Embedder:
    def embed(self, texts):
        return [[3, 4] for _ in texts]


def test_embeddings_are_separate_indexed_standard_npy(semantic_wiki, tmp_path):
    before = verify_wiki(semantic_wiki)
    output = tmp_path / "embeddings"
    report = export_embeddings(semantic_wiki, output, model="fixture", base_url="https://example.test/v1",
                               batch_size=2, client=Embedder())
    assert report["count"] == 3 and report["dimensions"] == 2
    assert report["wiki_hash"] == before["wiki_hash"]
    assert verify_wiki(semantic_wiki) == before
    with (output / "vectors.npy").open("rb") as stream:
        assert stream.read(8) == b"\x93NUMPY\x01\x00"
        header_size = struct.unpack("<H", stream.read(2))[0]
        header = ast.literal_eval(stream.read(header_size).decode().strip())
        assert header == {"descr": "<f4", "fortran_order": False, "shape": (3, 2)}
        assert struct.unpack("<6f", stream.read()) == pytest.approx([0.6, 0.8] * 3)
    assert [r["node_id"] for r in read_jsonl(output / "index.jsonl")] == [f"chapter_{i}" for i in range(3)]
    with pytest.raises(HarnessError, match="already exists"):
        export_embeddings(semantic_wiki, output, model="fixture", base_url="https://example.test", client=Embedder())
    with pytest.raises(HarnessError, match="outside"):
        export_embeddings(semantic_wiki, semantic_wiki / "embeddings", model="fixture",
                          base_url="https://example.test", client=Embedder())


@pytest.mark.parametrize("vectors", [[[0, 0]] * 3, [[True, 1]] * 3, [[float('nan'), 1]] * 3,
                                     [[1, 2], [1], [1, 2]], [[1, 2]]])
def test_invalid_embeddings_never_publish(semantic_wiki, tmp_path, vectors):
    class Invalid:
        def embed(self, texts):
            return vectors
    output = tmp_path / "embeddings"
    with pytest.raises(HarnessError):
        export_embeddings(semantic_wiki, output, model="fixture", base_url="https://example.test", client=Invalid())
    assert not output.exists()
    assert not list(tmp_path.glob(".embeddings-*"))


def test_embedding_api_reorders_indices_and_rejects_duplicates(monkeypatch):
    monkeypatch.setenv("EMBEDDING_KEY", "fixture")
    results = [{"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]},
               {"data": [{"index": 0, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}]
    def request(req, timeout):
        assert json.loads(req.data)["encoding_format"] == "float"
        return io.BytesIO(json.dumps(results.pop(0)).encode())
    monkeypatch.setattr("urllib.request.urlopen", request)
    client = EmbeddingClient("fixture", "https://example.test/v1", "EMBEDDING_KEY")
    assert client.embed(["one", "two"]) == [[1, 0], [0, 1]]
    with pytest.raises(HarnessError, match="indices"):
        client.embed(["one", "two"])
