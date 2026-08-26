import torch
from nn_compression.compression import (
    build_tucker2_conv,
    build_tucker2_conv_from_components,
    hooi,
    hosvd,
    reconstruct_tucker,
    tucker2_decompose_conv_weight,
    tucker2_hooi,
)
from nn_compression.metrics import relative_frobenius_error

print("=" * 60)
print("1. 3階テンソル HOOI (6,5,4), ranks={0:3,1:2,2:2}")
print("=" * 60)
torch.manual_seed(0)
X = torch.randn(6, 5, 4)
ranks = {0: 3, 1: 2, 2: 2}
core_h, f_h = hosvd(X, ranks)
err_h = float(relative_frobenius_error(X, reconstruct_tucker(core_h, f_h)))
core, factors, hist = hooi(X, ranks, max_iter=10)
print(f"  HOSVD error : {err_h:.6f}")
print(f"  HOOI  error : {hist[-1]:.6f}")
print(f"  improvement : {err_h - hist[-1]:.6f}")
print(f"  core shape  : {tuple(core.shape)}")
print(f"  iterations  : {len(hist) - 1}")
print(f"  history     : {[round(h, 6) for h in hist]}")

print()
print("=" * 60)
print("2. Conv weight Tucker-2 HOOI (64,32,3,3), rank_out=32, rank_in=16")
print("=" * 60)
torch.manual_seed(0)
W = torch.randn(64, 32, 3, 3)
ro, ri = 32, 16
core_h, f_h = hosvd(W, {0: ro, 1: ri})
err_h = float(relative_frobenius_error(W, reconstruct_tucker(core_h, f_h)))
core, factors, hist = tucker2_hooi(W, ro, ri, max_iter=10)
print(f"  HOSVD error : {err_h:.6f}")
print(f"  HOOI  error : {hist[-1]:.6f}")
print(f"  improvement : {err_h - hist[-1]:.6f}")
print(f"  core shape  : {tuple(core.shape)}")
print(f"  U_out shape : {tuple(factors[0].shape)}")
print(f"  U_in  shape : {tuple(factors[1].shape)}")
print(f"  iterations  : {len(hist) - 1}")
print(f"  history     : {[round(h, 6) for h in hist]}")

print()
print("=" * 60)
print("3. build_tucker2_conv vs build_tucker2_conv_from_components")
print("=" * 60)
from torch import nn
conv = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
torch.manual_seed(0)
with torch.no_grad():
    conv.weight.normal_()
    conv.bias.normal_()
seq1 = build_tucker2_conv(conv, ro, ri)
core, u_out, u_in = tucker2_decompose_conv_weight(conv.weight.detach(), ro, ri)
seq2 = build_tucker2_conv_from_components(conv, core, u_out, u_in)
x = torch.randn(2, 32, 8, 8)
with torch.no_grad():
    y1, y2 = seq1(x), seq2(x)
fwd_err = float(relative_frobenius_error(y1, y2))
print(f"  forward output relative error : {fwd_err:.2e}")
print(f"  match                         : {fwd_err < 1e-5}")

print()
print("ALL OK")
