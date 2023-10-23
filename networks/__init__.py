from .conn_snn import ConnSNN

from .dense_snn import DenseSNN

from .dense_mlp import DenseMLP
from .dense_gru import DenseGRU
from .dense_lstm import DenseLSTM


NETWORKS = {
    "ConnSNN": ConnSNN,

    # Dense
    "DenseSNN": DenseSNN,

    "DenseMLP": DenseMLP,
    "DenseGRU": DenseGRU,
    "DenseLSTM": DenseLSTM
}
