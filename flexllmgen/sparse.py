import dataclasses

@dataclasses.dataclass
class SparseConfig:
    """ sparse config"""
    mode: str
    block_size: int = 64
    sparsity: float = 0.6



