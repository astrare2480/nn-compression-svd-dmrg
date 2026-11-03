"""GpuResidentTensorBatches の公開 import が定義元と同一であること。"""

from nn_compression.training import GpuResidentTensorBatches as exported
from nn_compression.training.gpu_resident_batches import (
    GpuResidentTensorBatches as defined,
)


def test_package_exports_the_defining_class():
    assert exported is defined
