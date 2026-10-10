"""圧縮前後のモデル・テンソルを比較する評価指標・計算量推定。"""

from .cnn_macs import (
    estimate_cnn_conv2_macs,
    estimate_cnn_linear_macs,
    estimate_cnn_macs,
)
from .macs import (
    compressed_conv2d_macs,
    compressed_linear_macs,
    conv2d_macs,
    estimate_conv2d_macs,
    factorized_conv2d_macs,
    factorized_linear_macs,
    linear_macs,
)
from .error_bounds import (
    LinearOutputErrorBounds,
    linear_output_error_bounds,
    mlp_logits_error_bound,
    mlp_sample_logits_error_bounds,
)
from .mlp_macs import estimate_mlp_macs
from .model_comparison import (
    accuracy_drop,
    agreement,
    benchmark_inference,
    benchmark_inference_print,
    collect_compression_metrics,
    take_inference_batch,
    count_parameters,
    logits_rmse,
    parameters_reduction,
)
from .prediction_stability import (
    BatchStabilityCertificate,
    MinimumMargin,
    batch_stability_certificate,
    classification_margins,
    minimum_classification_margin,
    sample_stability_certificates,
)
from .tensor_approximation import relative_frobenius_error

__all__ = [
    "count_parameters",
    "parameters_reduction",
    "accuracy_drop",
    "agreement",
    "logits_rmse",
    "linear_macs",
    "compressed_linear_macs",
    "factorized_linear_macs",
    "conv2d_macs",
    "compressed_conv2d_macs",
    "estimate_conv2d_macs",
    "factorized_conv2d_macs",
    "estimate_mlp_macs",
    "estimate_cnn_macs",
    "estimate_cnn_linear_macs",
    "estimate_cnn_conv2_macs",
    "benchmark_inference",
    "benchmark_inference_print",
    "collect_compression_metrics",
    "take_inference_batch",
    "relative_frobenius_error",
    "LinearOutputErrorBounds",
    "linear_output_error_bounds",
    "mlp_logits_error_bound",
    "mlp_sample_logits_error_bounds",
    "MinimumMargin",
    "BatchStabilityCertificate",
    "classification_margins",
    "minimum_classification_margin",
    "sample_stability_certificates",
    "batch_stability_certificate",
]
