"""04_main_experiment_solver_dev.ipynb を作る。実行後は残さない。"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELLS = []


def md(source: str) -> None:
    CELLS.append(new_markdown_cell(source.strip() + "\n"))


def code(source: str) -> None:
    CELLS.append(new_code_cell(source.strip() + "\n"))


md(
    """
# 04 ソルバー開発: プロファイル、Gram、等価性、τ_warn の事前確認

主実験のノートはまだ作らない。このノートは A → E → F → G の順で、ソルバーの測定と確認だけを行う。
`cg_check_every` は 10 に固定し、結果を見て変えない。τ_warn もここでは選ばない。
"""
)

md(
    """
## 0. 設定

- 入力: プロジェクト直下、予備実験と 03 の保存ファイル。
- このステップで行うこと: スレッド数、オフラインの Hugging Face、出力先を決める。
- 行わないこと: モデルの読み込み、ALS、τ_warn の選択。
- 出力: `CONFIG`、`RESULTS`、`MODELS`、`DATA_DIR`。
- 完了条件: 03 の設定（float32_refine、Jacobi、fixed、相対許容誤差 1e-8）が印字される。
- 次のステップへの接続: 1 はこの設定で、掃引 1000 のコアをプロファイルする。
"""
)

code(
    r'''
import gc
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PROJECT = Path(r"D:\dev\nn-compression-svd-dmrg")
os.environ.setdefault("HF_HOME", str(PROJECT / "data" / "hf_cache"))
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(PROJECT)

import torch

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from nn_compression.utils.artifacts import atomic_torch_save, atomic_write_csv, atomic_write_json, file_sha256
from nn_compression.utils.paths import get_experiment_dirs
from nn_compression.tt_mps.metric_tt_solver import ConvergenceStopRule, MetricTTSolverConfig

if not torch.cuda.is_available():
    raise RuntimeError("CUDA がありません。")

CONFIG = MetricTTSolverConfig(
    device="cuda",
    precision="float32_refine",
    preconditioner="jacobi",
    cg_tolerance_mode="fixed",
    cg_relative_tolerance=1e-8,
    max_cg_iterations=500,
    max_refinements=5,
)
SMOKE_RULE = ConvergenceStopRule(max_sweeps=5)
_pilot_data, PILOT_MODELS, PILOT_RESULTS = get_experiment_dirs(
    PROJECT, "50_llm_tt_vs_matrix", "10_qwen3_0p6b_base", "01_single_layer_pilot", create=False
)
_long_data, LONG_MODELS, LONG_RESULTS = get_experiment_dirs(
    PROJECT, "50_llm_tt_vs_matrix", "10_qwen3_0p6b_base", "03_long_als_convergence", create=False
)
PILOT_DATA = _pilot_data / "50_llm_tt_vs_matrix" / "10_qwen3_0p6b_base" / "01_single_layer_pilot"
_, MODELS, RESULTS = get_experiment_dirs(
    PROJECT, "50_llm_tt_vs_matrix", "10_qwen3_0p6b_base", "04_main_experiment", create=True
)
MODELS = MODELS / "solver_dev"
RESULTS = RESULTS / "solver_dev"
DATA_DIR = PROJECT / "data" / "50_llm_tt_vs_matrix" / "10_qwen3_0p6b_base" / "04_main_experiment"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
GPU_ROWS = []
SETTINGS = {
    "config": asdict(CONFIG),
    "cg_check_every_equivalence": [1, 10],
    "tau_warn_provisional": 1e-7,
    "equivalence_q_tolerance": 1e-6,
    "environment": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    },
}


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf"
        return float(value)
    raise TypeError(f"JSON にできない型です: {type(value)!r}")


def write_json(path, payload):
    atomic_write_json(path, json_ready(payload))


def compute_apps():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    rows = []
    if text:
        for line in text.splitlines():
            pid, name, memory = [part.strip() for part in line.split(",", 2)]
            rows.append({"pid": int(pid), "process_name": name, "used_gpu_memory": memory})
    return rows


class GpuSampler:
    def __init__(self, phase):
        self.phase = phase
        self.rows = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            text = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            util, memory = [part.strip() for part in text.split(",")]
            self.rows.append(
                {
                    "phase": self.phase,
                    "elapsed_seconds": time.perf_counter() - self._t0,
                    "utilization_gpu": int(util),
                    "memory_used_mib": int(memory),
                }
            )
            self._stop.wait(1.0)

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        GPU_ROWS.extend(self.rows)
        atomic_write_csv(
            RESULTS / "gpu_utilization.csv",
            GPU_ROWS,
            ["phase", "elapsed_seconds", "utilization_gpu", "memory_used_mib"],
        )


print("CONFIG", asdict(CONFIG), flush=True)
print("GPU", SETTINGS["environment"]["gpu"], flush=True)
'''
)

md(
    """
## 1. プロファイル

- 入力: `long_als_T2_50_M3.pt`（掃引 1000 の終わり、中心 0）、予備実験の up_proj の重みと G_λ。
- このステップで行うこと: サイト 1 の 1 更新を CPU と CUDA で分け、その後 20 掃引の GPU 使用率を 1 秒ごとに取る。
- 行わないこと: `cg_check_every=10`、CUDA Graph、TF32、許容誤差の変更。大きいトレースは保存しない。
- 出力: `profile.json`、`gpu_utilization.csv`。
- 完了条件: `.item()` の待ち時間が、HVP 以外の時間の過半かどうかが数値で残っている。
- 次のステップへの接続: 2 は同じ up_proj の較正データで Gram を作り直す。
"""
)

code(
    r'''
from torch.profiler import ProfilerActivity, profile

from nn_compression.compression.metric_low_rank import regularize_gram
from nn_compression.tt_mps.matrix_two_site import canonicalize_tt_matrix_cores
from nn_compression.tt_mps.metric_tt import move_tt_matrix_center
from nn_compression.tt_mps.metric_tt_solver import (
    _Accumulator,
    _matmul_without_tf32,
    _solve_site,
    run_metric_tt_als_on_device,
)

APPS_BEFORE = compute_apps()
OTHER_PYTHON = [
    row for row in APPS_BEFORE
    if row["pid"] != os.getpid() and "python" in row["process_name"].lower()
]
idle_samples = []
for _ in range(3):
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    idle_samples.append(int(text))
    time.sleep(1.0)
# WDDM ではデスクトップのプロセスも並ぶ。別の計算だとみなすのは、他の Python か、開始前の使用率が 30% を超えるとき。
SETTINGS["gpu_process_count"] = len(APPS_BEFORE)
SETTINGS["other_python_processes"] = OTHER_PYTHON
SETTINGS["idle_gpu_utilization"] = idle_samples
SETTINGS["gpu_contended"] = bool(OTHER_PYTHON) or any(sample > 30 for sample in idle_samples)
print(
    "gpu processes", len(APPS_BEFORE),
    "other python", OTHER_PYTHON,
    "idle util", idle_samples,
    "contended", SETTINGS["gpu_contended"],
    flush=True,
)

checkpoint_path = LONG_MODELS / "long_als_T2_50_M3.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint["sweeps_completed"] != 1000 or checkpoint["stop_reason"] != "max_sweeps":
    raise RuntimeError(
        f"up の開始状態が掃引 1000・max_sweeps ではありません: "
        f"sweeps={checkpoint['sweeps_completed']} stop={checkpoint['stop_reason']}"
    )
m3_payload = torch.load(PILOT_MODELS / "m3_T2_50.pt", map_location="cpu", weights_only=False)
OUT_MODES = tuple(int(mode) for mode in m3_payload["out_modes"])
IN_MODES = tuple(int(mode) for mode in m3_payload["in_modes"])
weights = torch.load(PILOT_DATA / "weights_float64.pt", map_location="cpu", weights_only=False)
W = weights["W"].to(dtype=torch.float64, device="cpu").contiguous()
calibration = torch.load(PILOT_DATA / "calibration_grams.pt", map_location="cpu", weights_only=False)
calibration.pop("per_document", None)
G = calibration["total"].to(dtype=torch.float64, device="cpu").contiguous()
G_LAMBDA = regularize_gram(G, 1e-6)
gc.collect()

device = torch.device("cuda")
profile_cores = [core.to(device=device, dtype=torch.float64) for core in checkpoint["cores"]]
profile_cores = canonicalize_tt_matrix_cores(profile_cores, OUT_MODES, IN_MODES, 0)
profile_cores = move_tt_matrix_center(profile_cores, OUT_MODES, IN_MODES, 0, 1)
gram64 = G_LAMBDA.to(device=device, dtype=torch.float64)
gram32 = gram64.to(dtype=torch.float32)
weight64 = W.to(device=device, dtype=torch.float64)
weight_gram = weight64 @ gram64
hvp_time = _Accumulator(device)
pre_time = _Accumulator(device)
apply_time = _Accumulator(device)
statuses = []
torch.cuda.synchronize()
wall_started = time.perf_counter()
with _matmul_without_tf32():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        _solve_site(
            profile_cores,
            1,
            gram64,
            gram32,
            weight_gram,
            OUT_MODES,
            IN_MODES,
            CONFIG,
            {},
            hvp_time,
            pre_time,
            apply_time,
            cg_statuses=statuses,
        )
torch.cuda.synchronize()
wall_seconds = time.perf_counter() - wall_started


def event_seconds(event, *names):
    for name in names:
        if hasattr(event, name):
            return float(getattr(event, name)) / 1e6
    return 0.0


grouped = []
for event in prof.key_averages():
    key = event.key
    if not any(token in key for token in ("hvp_", "cg_", "item", "Synchronize", "qr", "dot", "norm", "mm")):
        continue
    grouped.append(
        {
            "key": key,
            "count": int(event.count),
            "cpu_time_total": event_seconds(event, "cpu_time_total"),
            "self_cpu_time_total": event_seconds(event, "self_cpu_time_total"),
            "device_time_total": event_seconds(event, "device_time_total", "cuda_time_total"),
            "self_device_time_total": event_seconds(event, "self_device_time_total", "self_cuda_time_total"),
        }
    )
item_cpu = sum(
    row["cpu_time_total"] for row in grouped if "cg_curvature_item" in row["key"] or "cg_residual_item" in row["key"]
)
hvp_cuda = sum(
    row["device_time_total"] for row in grouped if any(name in row["key"] for name in ("hvp_dense", "hvp_gemm", "hvp_adjoint"))
)
if hvp_cuda == 0.0:
    hvp_cuda = sum(
        row["self_device_time_total"]
        for row in grouped
        if any(name in row["key"] for name in ("hvp_dense", "hvp_gemm", "hvp_adjoint", "mm", "addmm", "bmm", "einsum"))
    )
non_hvp = wall_seconds - hvp_cuda
hypothesis_holds = item_cpu >= 0.5 * non_hvp if non_hvp > 0 else False
del prof
print(
    f"site1 wall {wall_seconds:.3f}s item_cpu {item_cpu:.3f}s hvp_cuda {hvp_cuda:.3f}s "
    f"non_hvp {non_hvp:.3f}s hypothesis {hypothesis_holds} statuses {statuses}",
    flush=True,
)
for row in sorted(grouped, key=lambda item: item["cpu_time_total"], reverse=True)[:12]:
    print(
        f"  {row['key']} count {row['count']} cpu {row['cpu_time_total']:.3f}s cuda {row['device_time_total']:.3f}s",
        flush=True,
    )

# このコアはすでに掃引 1000 まで進んでおり、既定の sweep_tolerance=1e-6 だと 1 掃引で止まる。
# 関数の既定は変えず、この測定だけ負の値にして 20 掃引を最後まで回す。
sweep_config = replace(CONFIG, max_sweeps=20, sweep_tolerance=-1.0)
with GpuSampler("profile_20_sweeps"):
    _cores20, history20 = run_metric_tt_als_on_device(
        checkpoint["cores"], W, G_LAMBDA, OUT_MODES, IN_MODES, sweep_config,
        progress=lambda record: print(f"profile sweep {record.sweep} {record.seconds:.3f}s", flush=True),
    )
sweep_seconds = [record.seconds for record in history20]
hvp_seconds = [record.seconds_hvp for record in history20]
median_seconds = sorted(sweep_seconds)[len(sweep_seconds) // 2]
median_hvp = sorted(hvp_seconds)[len(hvp_seconds) // 2]
profile_payload = {
    "hypothesis": "掃引時間のうち HVP 以外の約 1.35 秒は、主に CG 内の .item() 同期である。",
    "hypothesis_rule": "サイト 1 の 1 更新で、cg_curvature_item と cg_residual_item の CPU 時間が、壁時計から HVP の CUDA 時間を引いた残りのうち 50% 以上なら成り立つ。",
    "hypothesis_holds": hypothesis_holds,
    "site1_wall_seconds": wall_seconds,
    "site1_item_cpu_seconds": item_cpu,
    "site1_hvp_cuda_seconds": hvp_cuda,
    "site1_non_hvp_seconds": non_hvp,
    "site1_preconditioner_seconds": pre_time.total,
    "site1_preconditioner_apply_seconds": apply_time.total,
    "site1_hvp_timer_seconds": hvp_time.total,
    "site1_cg_statuses": list(statuses),
    "sweep20_median_seconds": median_seconds,
    "sweep20_median_hvp_seconds": median_hvp,
    "sweep20_median_non_hvp_seconds": median_seconds - median_hvp,
    "sweep20_median_qr_seconds": sorted(record.seconds_qr for record in history20)[len(history20) // 2],
    "sweep20_median_accept_seconds": sorted(record.seconds_accept for record in history20)[len(history20) // 2],
    "sweep20_median_preconditioner_seconds": sorted(record.seconds_preconditioner for record in history20)[len(history20) // 2],
    "sweep20_median_preconditioner_apply_seconds": sorted(record.seconds_preconditioner_apply for record in history20)[len(history20) // 2],
    "events": grouped,
    "gpu_contended": SETTINGS["gpu_contended"],
}
write_json(RESULTS / "profile.json", profile_payload)
write_json(RESULTS / "settings.json", SETTINGS)
print("profile written", flush=True)
del checkpoint, profile_cores, gram64, gram32, weight64, weight_gram
torch.cuda.empty_cache()
gc.collect()
'''
)

md(
    """
## 2. 較正 Gram と D_ref

- 入力: WikiText-103 train、seed 0、128 系列 × 1024。予備実験の `data_preparation.json` と `calibration_grams.pt`。
- このステップで行うこと: 同じ順伝播で up_proj と down_proj の Gram を足す。down の T2・50% の M3 と、両方の D_ref を計算する。
- 行わないこと: 活性化の保存、検証データ、ALS。
- 出力: `data_checks.json`、Gram、`W_down`、down の M3。
- 完了条件: 文書 hash が同じ順で、up の Gram の相対フロベニウス差が 1e-12 以下。mode と rank が設計の表と一致する。
- 次のステップへの接続: 3 はこの D_ref で 50 掃引の等価性を見る。差が 1e-12 を超えたらここで止める。
"""
)

code(
    r'''
from nn_compression.compression.metric_low_rank import max_matrix_rank_within_budget, whitened_truncated_svd
from nn_compression.llm.activation_capture import accumulate_input_grams
from nn_compression.llm.hf_loading import load_causal_lm_and_tokenizer, load_download_record, load_tokenizer, load_wikitext
from nn_compression.llm.text_documents import restore_wikitext_articles, select_token_prefix_sequences
from nn_compression.tt_mps.matrix import dense_to_tt_matrix_cores
from nn_compression.tt_mps.metric_tt import q_lambda_of_cores, select_balanced_tt_matrix_modes, select_uniform_tt_rank

LAMBDA = 1e-6
LAYER_INDEX = 14


def cached_encoder(tokenizer, documents, batch_size=8):
    cache = {}
    total = len(documents)
    started = time.perf_counter()
    for start in range(0, total, batch_size):
        batch = list(documents[start : start + batch_size])
        try:
            encoded = tokenizer(
                batch, add_special_tokens=False, truncation=False, padding=False, return_attention_mask=False
            )
            token_rows = encoded["input_ids"]
        except Exception:
            token_rows = [tokenizer.encode(text, add_special_tokens=False, truncation=False) for text in batch]
        for text, token_ids in zip(batch, token_rows):
            if torch.is_tensor(token_ids):
                token_ids = token_ids.tolist()
            cache[text] = token_ids
        done = min(start + batch_size, total)
        if start == 0 or done == total or done % (batch_size * 100) == 0:
            print(f"  tokenized {done}/{total} ({time.perf_counter() - started:.1f}s)", flush=True)

    def encode(text):
        return cache[text]

    return encode


record = load_download_record(PROJECT / "data" / "hf_cache" / "download_record.json")
tokenizer = load_tokenizer(record)
print("WikiText-103 train を読みます。", flush=True)
wikitext = load_wikitext(record, "wikitext-103-raw-v1", "train")
articles = restore_wikitext_articles(list(wikitext["text"]))
print(f"articles {len(articles)}", flush=True)
selection = select_token_prefix_sequences(
    articles, cached_encoder(tokenizer, articles), source="wikitext-103-train",
    min_tokens=1024, length=1024, limit=128, seed=0,
)
del wikitext, articles
gc.collect()
prepared = json.loads((PILOT_RESULTS / "data_preparation.json").read_text(encoding="utf-8"))
expected_hashes = prepared["splits"]["calibration"]["document_hashes"]
got_hashes = [sequence.document_hash for sequence in selection.sequences]
hash_match = got_hashes == expected_hashes
print("hash match", hash_match, "n", len(got_hashes), flush=True)
if not hash_match:
    write_json(RESULTS / "data_checks.json", {"hash_match": False, "n_got": len(got_hashes), "n_expected": len(expected_hashes)})
    raise RuntimeError("較正文書の hash が予備実験と一致しません。")

print("モデルを読みます。", flush=True)
model, _loaded = load_causal_lm_and_tokenizer(record, dtype=torch.bfloat16, device="cpu")
if len(model.model.layers) != 28:
    raise RuntimeError(f"層数が {len(model.model.layers)} です。")
down = model.model.layers[LAYER_INDEX].mlp.down_proj
if down.bias is not None:
    raise RuntimeError("down_proj に bias があります。")
W_DOWN = down.weight.detach().to(device="cpu", dtype=torch.float64).contiguous()
if W_DOWN.shape != (1024, 3072):
    raise RuntimeError(f"W_down の shape が {tuple(W_DOWN.shape)} です。")
model = model.to(dtype=torch.float32, device="cuda")
model.eval()
up = model.model.layers[LAYER_INDEX].mlp.up_proj
down = model.model.layers[LAYER_INDEX].mlp.down_proj
sequences = [torch.tensor(sequence.token_ids, dtype=torch.long) for sequence in selection.sequences]
print("Gram を蓄積します。", flush=True)
grams = accumulate_input_grams(model, {"up_proj": up, "down_proj": down}, sequences, device="cuda")
del model, sequences
torch.cuda.empty_cache()
gc.collect()
up_total = grams["up_proj"]["total"]
up_gap = torch.linalg.matrix_norm(up_total - G) / torch.linalg.matrix_norm(G)
up_bitwise = bool(torch.equal(up_total, G))
print(f"up gram relative {float(up_gap)} bitwise {up_bitwise}", flush=True)

down_modes = select_balanced_tt_matrix_modes(1024, 3072, 3)
down_out = tuple(down_modes["out_modes"])
down_in = tuple(down_modes["in_modes"])
down_budget = 1024 * 3072 // 2
down_rank = select_uniform_tt_rank(down_out, down_in, down_budget)
down_ranks = tuple(int(rank) for rank in down_rank["ranks"])
modes_ok = down_out == (2, 8, 64) and down_in == (64, 16, 3) and down_ranks == (1, 109, 109, 1)
print("down modes", down_out, down_in, down_ranks, "ok", modes_ok, flush=True)

up_rank = max_matrix_rank_within_budget(3072, 1024, 3072 * 1024 // 2)
up_m2 = whitened_truncated_svd(W, G_LAMBDA, up_rank)
up_m2_abs = float((up_m2.q_lambda - up_m2.discarded_singular_value_energy).abs())
up_m2_rel = up_m2_abs / max(abs(float(up_m2.q_lambda)), 1.0)
up_m2_bitwise = bool(torch.equal(up_m2.q_lambda, up_m2.discarded_singular_value_energy))
up_m2_ok = up_m2_rel <= 1e-12
with (PILOT_RESULTS / "m5_initial_values.csv").open(encoding="utf-8", newline="") as handle:
    up_rows = [
        row for row in csv.DictReader(handle)
        if row["calibration_set"] == "full" and row["tensorization"] == "T2" and row["budget"] == "50" and row["initial"] == "M3"
    ]
if len(up_rows) != 1:
    raise RuntimeError(f"up の M3 参照が {len(up_rows)} 行です。")
q_up_m3 = float(up_rows[0]["q_lambda_before_als"])
q_up_m2 = float(up_m2.q_lambda)
d_ref_up = max(q_up_m3 - q_up_m2, 0.05 * q_up_m3)

down_lambda = regularize_gram(grams["down_proj"]["total"], LAMBDA)
down_m2_rank = max_matrix_rank_within_budget(1024, 3072, down_budget)
down_m2 = whitened_truncated_svd(W_DOWN, down_lambda, down_m2_rank)
down_m2_abs = float((down_m2.q_lambda - down_m2.discarded_singular_value_energy).abs())
down_m2_rel = down_m2_abs / max(abs(float(down_m2.q_lambda)), 1.0)
down_m2_bitwise = bool(torch.equal(down_m2.q_lambda, down_m2.discarded_singular_value_energy))
down_m2_ok = down_m2_rel <= 1e-12
down_cores = dense_to_tt_matrix_cores(W_DOWN, down_out, down_in, max_rank=down_ranks[1])
q_down_m3 = float(q_lambda_of_cores(down_cores, W_DOWN, down_lambda, down_out, down_in))
q_down_m2 = float(down_m2.q_lambda)
d_ref_down = max(q_down_m3 - q_down_m2, 0.05 * q_down_m3)

checks = {
    "hash_match": hash_match,
    "n_sequences": len(got_hashes),
    "up_n_tokens": grams["up_proj"]["n_tokens"],
    "down_n_tokens": grams["down_proj"]["n_tokens"],
    "up_gram_relative_frobenius": float(up_gap),
    "up_gram_bitwise_equal": up_bitwise,
    "down_out_modes": list(down_out),
    "down_in_modes": list(down_in),
    "down_ranks": list(down_ranks),
    "modes_and_ranks_match_design": modes_ok,
    "up_m2_abs_gap": up_m2_abs,
    "up_m2_relative_gap": up_m2_rel,
    "up_m2_bitwise_equal": up_m2_bitwise,
    "up_m2_within_1e-8": up_m2_abs <= 1e-8,
    "up_m2_ok": up_m2_ok,
    "down_m2_abs_gap": down_m2_abs,
    "down_m2_relative_gap": down_m2_rel,
    "down_m2_bitwise_equal": down_m2_bitwise,
    "down_m2_within_1e-8": down_m2_abs <= 1e-8,
    "down_m2_ok": down_m2_ok,
    "up_m2_rank": int(up_rank),
    "down_m2_rank": int(down_m2_rank),
    "q_up_m3": q_up_m3,
    "q_up_m2": q_up_m2,
    "d_ref_up": d_ref_up,
    "q_down_m3": q_down_m3,
    "q_down_m2": q_down_m2,
    "d_ref_down": d_ref_down,
    "up_small_baseline_gap": (q_up_m3 - q_up_m2) < 0.05 * q_up_m3,
    "down_small_baseline_gap": (q_down_m3 - q_down_m2) < 0.05 * q_down_m3,
}
write_json(RESULTS / "data_checks.json", checks)
write_json(DATA_DIR / "calibration_document_hashes.json", got_hashes)
atomic_torch_save(
    DATA_DIR / "calibration_grams.pt",
    {
        "up_proj_total": up_total,
        "up_proj_n_tokens": grams["up_proj"]["n_tokens"],
        "down_proj_total": grams["down_proj"]["total"],
        "down_proj_n_tokens": grams["down_proj"]["n_tokens"],
    },
)
atomic_torch_save(DATA_DIR / "w_down_float64.pt", {"W_down": W_DOWN})
atomic_torch_save(
    MODELS / "m3_down_T2_50.pt",
    {"cores": down_cores, "out_modes": down_out, "in_modes": down_in, "ranks": down_ranks},
)
print(
    f"D_ref up {d_ref_up} down {d_ref_down} m2 abs up {up_m2_abs} down {down_m2_abs}",
    flush=True,
)
if not (hash_match and modes_ok and up_m2_ok and down_m2_ok):
    raise RuntimeError("hash、mode、rank、または M2 の検算が通りません。")
if float(up_gap) > 1e-12:
    raise RuntimeError(f"up の Gram の相対差が 1e-12 を超えています: {float(up_gap)}")
D_REF_UP = d_ref_up
D_REF_DOWN = d_ref_down
DOWN_CORES = down_cores
DOWN_OUT = down_out
DOWN_IN = down_in
DOWN_LAMBDA = down_lambda
DOWN_RAW = grams["down_proj"]["total"]
SETTINGS["d_ref_up"] = D_REF_UP
SETTINGS["d_ref_down"] = D_REF_DOWN
SETTINGS["input_sha256"] = {
    "long_als_T2_50_M3": file_sha256(LONG_MODELS / "long_als_T2_50_M3.pt"),
    "long_als_T2_50_M4_retruncated": file_sha256(LONG_MODELS / "long_als_T2_50_M4_retruncated.pt"),
    "pilot_calibration_grams": file_sha256(PILOT_DATA / "calibration_grams.pt"),
    "pilot_weights": file_sha256(PILOT_DATA / "weights_float64.pt"),
    "history_smoke": file_sha256(LONG_RESULTS / "history_smoke.csv"),
}
write_json(RESULTS / "settings.json", SETTINGS)
'''
)

md(
    """
## 3. 等価性

- 入力: 予備実験の up M3、このノートで作った down M3、D_ref、03 の `history_smoke.csv`。
- このステップで行うこと: 旧関数で up を 5 掃引し、新しい関数で up と down を k=1 と k=10 で 50 掃引ずつ比べる。
- 行わないこと: k を変えてやり直すこと、主実験の他の層。
- 出力: `equivalence.json`、`equivalence_history.csv`、チェックポイント。
- 完了条件: 時間以外の 5 掃引が smoke とビット単位で一致し、50 掃引の区分が同じで |ΔQ| / D_ref が 1e-6 以下。
- 次のステップへの接続: 4 はこの up の D_ref で、03 が止まった局所問題を 5 掃引見る。不合格ならここで止める。
"""
)

code(
    r'''
from nn_compression.tt_mps.metric_tt_solver import StudyResolutionStopRule, run_metric_tt_als_study, run_metric_tt_als_until_converged

UP_CORES = [core.detach().to(dtype=torch.float64, device="cpu").contiguous() for core in m3_payload["cores"]]


def parse_bool_list(text):
    if text == "":
        return []
    return [part == "True" for part in text.split(";")]


def parse_float_list(text):
    if text == "":
        return []
    return [float(part) for part in text.split(";")]


def parse_int_list(text):
    if text == "":
        return []
    return [int(part) for part in text.split(";")]


smoke_rows = []
with (LONG_RESULTS / "history_smoke.csv").open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["budget"] == "50" and row["initial"] == "M3":
            smoke_rows.append(row)
print("F.1 を回します。", flush=True)
f1 = run_metric_tt_als_until_converged(
    UP_CORES, W, G_LAMBDA, OUT_MODES, IN_MODES, CONFIG, SMOKE_RULE, report_gram=G,
    progress=lambda record: print(f"F1 sweep {record.sweep} Q {record.q_lambda:.8g}", flush=True),
)
f1_fields = []
f1_ok = len(f1.history) == len(smoke_rows) == 5
mismatches = []
for record, row in zip(f1.history, smoke_rows):
    compared = {
        "q_lambda": (record.q_lambda, float(row["q_lambda"])),
        "q_unregularized": (record.q_unregularized, float(row["q_unregularized"])),
        "relative_decrease": (record.relative_decrease, float(row["relative_decrease"])),
        "small_decrease_streak": (record.small_decrease_streak, int(row["small_decrease_streak"])),
        "monotone_ok": (record.monotone_ok, row["monotone_ok"] == "True"),
        "stationarity": (tuple(record.stationarity), tuple(float(row[name]) for name in ("s_0", "s_1", "s_2"))),
        "stationarity_max": (record.stationarity_max, float(row["stationarity_max"])),
        "update_sites": (list(record.update_sites), parse_int_list(row["update_sites"])),
        "cg_iterations": (list(record.cg_iterations), parse_int_list(row["cg_iterations"])),
        "cg_residuals": (list(record.cg_residuals), parse_float_list(row["cg_residuals"])),
        "refinements": (list(record.refinements), parse_int_list(row["refinements"])),
        "retried": (list(record.retried), parse_bool_list(row["retried"])),
        "accepted": (list(record.accepted), parse_bool_list(row["accepted"])),
        "n_rejected": (record.n_rejected, int(row["n_rejected"])),
        "stop_reason": (record.stop_reason, row["stop_reason"] or None),
    }
    for name, (got, want) in compared.items():
        if got != want:
            f1_ok = False
            mismatches.append({"sweep": record.sweep, "field": name, "got": got, "want": want})
print("F.1", f1_ok, "mismatches", len(mismatches), flush=True)


def study_rule(d_ref):
    return StudyResolutionStopRule(
        d_ref=d_ref, window_sweeps=50, resolution_tolerance=1e-4, consecutive_windows=2,
        max_sweeps=50, local_residual_tolerance=1e-8, escalation_refinements=(10, 50),
        warn_residual_tolerance=1e-8,
    )


def run_equivalence(name, cores, weight, gram, raw_gram, out_modes, in_modes, d_ref, check_every):
    print(f"F.2 {name} k={check_every}", flush=True)
    config = replace(CONFIG, cg_check_every=check_every, max_sweeps=50)
    with GpuSampler(f"equivalence_{name}_k{check_every}"):
        result = run_metric_tt_als_study(
            cores, weight, gram, out_modes, in_modes, config, study_rule(d_ref), report_gram=raw_gram,
            checkpoint_path=MODELS / f"equivalence_{name}_k{check_every}.pt",
            metadata={"projection": name, "cg_check_every": check_every, "budget": "50", "initial": "M3"},
            progress=lambda record: print(
                f"{name} k={check_every} sweep {record.sweep} Q {record.q_lambda:.8g} "
                f"{record.seconds:.2f}s stage3 {record.n_stage3}",
                flush=True,
            ),
        )
    return result


def flat_updates(result):
    rows = []
    for record in result.history:
        for update in record.updates:
            rows.append(
                {
                    "sweep": record.sweep,
                    "site": update.site,
                    "outcome": update.outcome,
                    "accepted": update.accepted,
                    "n_stages": len(update.stages),
                    "cg_iterations": sum(stage.cg_iterations for stage in update.stages),
                    "curvature_nonpositive": sum(stage.nonpositive_curvature_count for stage in update.stages),
                    "residual": update.stages[-1].residual,
                }
            )
    return rows


def job_summary(result):
    seconds = [record.seconds for record in result.history]
    hvp_calls = sum(record.hvp_calls for record in result.history)
    total_seconds = sum(seconds)
    updates = flat_updates(result)
    return {
        "stop_reason": result.stop_reason,
        "n_sweeps": len(result.history),
        "q_lambda": None if not result.history else result.history[-1].q_lambda,
        "seconds_per_sweep_mean": None if not seconds else total_seconds / len(seconds),
        "hvp_per_second": None if total_seconds == 0 else hvp_calls / total_seconds,
        "hvp_calls": hvp_calls,
        "curvature_nonpositive": sum(row["curvature_nonpositive"] for row in updates),
        "n_stage2": result.n_stage2,
        "n_stage3": result.n_stage3,
        "n_warnings": result.n_warnings,
        "outcomes": [row["outcome"] for row in updates],
        "accepted": [row["accepted"] for row in updates],
        "n_stages": [row["n_stages"] for row in updates],
    }


jobs = {}
if f1_ok:
    jobs["up_k1"] = run_equivalence("up", UP_CORES, W, G_LAMBDA, G, OUT_MODES, IN_MODES, D_REF_UP, 1)
    jobs["up_k10"] = run_equivalence("up", UP_CORES, W, G_LAMBDA, G, OUT_MODES, IN_MODES, D_REF_UP, 10)
    jobs["down_k1"] = run_equivalence("down", DOWN_CORES, W_DOWN, DOWN_LAMBDA, DOWN_RAW, DOWN_OUT, DOWN_IN, D_REF_DOWN, 1)
    jobs["down_k10"] = run_equivalence("down", DOWN_CORES, W_DOWN, DOWN_LAMBDA, DOWN_RAW, DOWN_OUT, DOWN_IN, D_REF_DOWN, 10)

summaries = {name: job_summary(result) for name, result in jobs.items()}
comparisons = {}
f2_ok = f1_ok and len(jobs) == 4
for name, d_ref in (("up", D_REF_UP), ("down", D_REF_DOWN)):
    if f"{name}_k1" not in summaries:
        f2_ok = False
        continue
    left = summaries[f"{name}_k1"]
    right = summaries[f"{name}_k10"]
    same_class = left["outcomes"] == right["outcomes"]
    q_gap = None
    q_ok = False
    if left["q_lambda"] is not None and right["q_lambda"] is not None and left["n_sweeps"] == right["n_sweeps"] == 50:
        q_gap = abs(right["q_lambda"] - left["q_lambda"]) / d_ref
        q_ok = q_gap <= 1e-6
    accept_differences = sum(a != b for a, b in zip(left["accepted"], right["accepted"]))
    stage_differences = sum(a != b for a, b in zip(left["n_stages"], right["n_stages"]))
    passed = same_class and q_ok and left["stop_reason"] == "max_sweeps" and right["stop_reason"] == "max_sweeps"
    comparisons[name] = {
        "same_class": same_class,
        "q_abs_gap_over_d_ref": q_gap,
        "q_ok": q_ok,
        "accept_differences": accept_differences,
        "stage_differences": stage_differences,
        "passed": passed,
    }
    f2_ok = f2_ok and passed

history_rows = []
for name, result in jobs.items():
    for record in result.history:
        history_rows.append(
            {
                "job": name,
                "sweep": record.sweep,
                "q_lambda": record.q_lambda,
                "seconds": record.seconds,
                "hvp_calls": record.hvp_calls,
                "n_stage2": record.n_stage2,
                "n_stage3": record.n_stage3,
                "n_warnings": record.n_warnings,
                "stop_reason": "" if record.stop_reason is None else record.stop_reason,
                "outcomes": ";".join(update.outcome for update in record.updates),
                "accepted": ";".join(str(update.accepted) for update in record.updates),
                "n_stages": ";".join(str(len(update.stages)) for update in record.updates),
                "cg_iterations": ";".join(str(sum(stage.cg_iterations for stage in update.stages)) for update in record.updates),
                "curvature_nonpositive": ";".join(
                    str(sum(stage.nonpositive_curvature_count for stage in update.stages)) for update in record.updates
                ),
            }
        )
if history_rows:
    atomic_write_csv(RESULTS / "equivalence_history.csv", history_rows, list(history_rows[0].keys()))

public_summaries = {}
for name, summary in summaries.items():
    public_summaries[name] = {key: value for key, value in summary.items() if key not in {"outcomes", "accepted", "n_stages"}}
    public_summaries[name]["n_updates"] = len(summary["outcomes"])
equivalence = {
    "f1_passed": f1_ok,
    "f1_mismatches": mismatches,
    "f2_passed": f2_ok,
    "comparisons": comparisons,
    "jobs": public_summaries,
}
write_json(RESULTS / "equivalence.json", equivalence)
print("F passed", f1_ok, f2_ok, flush=True)
if not f1_ok or not f2_ok:
    raise RuntimeError("等価性の条件を満たしていません。k は変更しません。")
'''
)

md(
    """
## 4. τ_warn の事前確認

- 入力: `long_als_T2_50_M4_retruncated.pt`（掃引 426 の終わり）、up の D_ref、仮の τ_warn = 1e-7。
- このステップで行うこと: `cg_check_every=10` で 5 掃引し、03 が止まった局所問題（この実行の掃引 1・サイト 1）の各段を記録する。
- 行わないこと: τ_warn を決めること、別の候補を試すこと。
- 出力: `tau_warn_precheck.json`。
- 完了条件: その局所問題の各段の残差、3 段目が 1e-7 以下か、確認値 a、時間が、03 の `failures.json` の残差と並んで残っている。
- 次のステップへの接続: 主実験のノートは、この結果を見てから別に作る。
"""
)

code(
    r'''
m4_path = LONG_MODELS / "long_als_T2_50_M4_retruncated.pt"
m4_state = torch.load(m4_path, map_location="cpu", weights_only=False)
if m4_state["sweeps_completed"] != 426 or m4_state["stop_reason"] != "local_solver_unresolved":
    raise RuntimeError(
        f"M4 の開始状態が掃引 426・local_solver_unresolved ではありません: "
        f"sweeps={m4_state['sweeps_completed']} stop={m4_state['stop_reason']}"
    )
failures = json.loads((LONG_RESULTS / "failures.json").read_text(encoding="utf-8"))
reference = next(item for item in failures if item["sweep"] == 427 and item["site"] == 1)
print("G を回します。", flush=True)
g_config = replace(CONFIG, cg_check_every=10, max_sweeps=5)
g_rule = StudyResolutionStopRule(
    d_ref=D_REF_UP, window_sweeps=50, max_sweeps=5, local_residual_tolerance=1e-8,
    escalation_refinements=(10, 50), warn_residual_tolerance=1e-7,
)
with GpuSampler("tau_warn_precheck"):
    g_result = run_metric_tt_als_study(
        m4_state["cores"], W, G_LAMBDA, OUT_MODES, IN_MODES, g_config, g_rule,
        checkpoint_path=MODELS / "tau_warn_precheck.pt",
        metadata={"purpose": "tau_warn_precheck", "source_sweep": 426, "cg_check_every": 10},
        progress=lambda record: print(
            f"G sweep {record.sweep} Q {record.q_lambda:.8g} warnings {record.n_warnings} stage3 {record.n_stage3}",
            flush=True,
        ),
    )


def stage_payload(stage):
    return {
        "stage": stage.stage,
        "max_refinements": stage.max_refinements,
        "residual": stage.residual,
        "cg_iterations": stage.cg_iterations,
        "refinements": stage.refinements,
        "seconds": stage.seconds,
        "cg_statuses": list(stage.cg_statuses),
        "nonpositive_curvature_count": stage.nonpositive_curvature_count,
        "q_lambda": stage.q_lambda,
    }


def update_payload(record, update):
    return {
        "sweep": record.sweep,
        "site": update.site,
        "outcome": update.outcome,
        "accepted": update.accepted,
        "confirmation_a": update.confirmation_a,
        "stages": [stage_payload(stage) for stage in update.stages],
    }


focus = None
escalated = []
if g_result.history:
    for record in g_result.history:
        for update in record.updates:
            if len(update.stages) >= 2:
                escalated.append(update_payload(record, update))
            if record.sweep == 1 and update.site == 1 and focus is None:
                focus = update_payload(record, update)
if focus is None and g_result.failure is not None and g_result.failure.sweep == 1 and g_result.failure.site == 1:
    focus = {
        "sweep": 1,
        "site": 1,
        "outcome": "local_solver_unresolved",
        "accepted": False,
        "confirmation_a": None,
        "stages": [stage_payload(stage) for stage in g_result.failure.stages],
        "q_lambda_before": g_result.failure.q_lambda_before,
        "q_lambda_candidate": g_result.failure.q_lambda_candidate,
    }
stage3_residual = None
stage3_reached_warn = None
if focus is not None and any(stage["stage"] == 3 for stage in focus["stages"]):
    stage3_residual = next(stage["residual"] for stage in focus["stages"] if stage["stage"] == 3)
    stage3_reached_warn = math.isfinite(stage3_residual) and stage3_residual <= 1e-7
precheck = {
    "source_sweeps_completed": 426,
    "source_stop_reason": "local_solver_unresolved",
    "provisional_tau_warn": 1e-7,
    "tau_warn_chosen": None,
    "run_stop_reason": g_result.stop_reason,
    "n_sweeps_completed": len(g_result.history),
    "focus_update": focus,
    "stage3_residual": stage3_residual,
    "stage3_residual_le_1e-7": stage3_reached_warn,
    "escalated_updates": escalated,
    "failures_json_sweep_427_site_1": {
        "first_residual": reference["first_residual"],
        "retry_residual": reference["retry_residual"],
        "cg_iterations": reference["cg_iterations"],
        "refinements": reference["refinements"],
    },
}
write_json(RESULTS / "tau_warn_precheck.json", precheck)
write_json(RESULTS / "settings.json", SETTINGS)
print("G stop", g_result.stop_reason, "stage3", stage3_residual, "le 1e-7", stage3_reached_warn, flush=True)
'''
)

notebook = new_notebook(
    cells=CELLS,
    metadata={"kernelspec": {"display_name": "pytorch-svd", "language": "python", "name": "pytorch-svd"}},
)
path = Path(__file__).with_name("04_main_experiment_solver_dev.ipynb")
nbformat.write(notebook, path)
print(path)
