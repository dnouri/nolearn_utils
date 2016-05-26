import pytest
import numpy as np
from nolearn_utils.npkvstore import NPKVStore
from nolearn_utils.utils import mkdirp


@pytest.fixture
def kv_store():
    mkdirp('/tmp/nolearn_utils/')
    kv_store = NPKVStore(
        '/tmp/nolearn_utils/npkvstore.dat',
        dtype=np.float32, mode='w+', shape=(100, 1, 28, 28)
    )
    return kv_store


def test_npkvstore_single_item(kv_store):
    kv_store['a', 1] = 42
    kv_store.flush()
    assert kv_store['a', 1] == 42
