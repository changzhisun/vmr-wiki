"""Legacy checkpoint location retained for existing unfinished builds."""
from vmr.compiler.checkpoint import CompileCheckpoint

class IngestCheckpoint(CompileCheckpoint):
    def __init__(self, output, identity):
        super().__init__(output, identity, namespace='.ingest-checkpoints')
