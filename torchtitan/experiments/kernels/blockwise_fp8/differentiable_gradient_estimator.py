import torch

device = torch.device("cpu")
hp_dtype = torch.float32
k = 5
BITS = {
    torch.float8_e4m3fnuz: 8,
    torch.float8_e5m2fnuz: 8,
}
INT_DTYPES = {
    torch.float8_e4m3fnuz: torch.uint8,
    torch.float8_e5m2fnuz: torch.uint8,
}


def calculate_break_points(dtype):
    bitwidth = BITS[dtype]
    int_dtype = INT_DTYPES[dtype]
    break_points = torch.arange(
        0, 2**bitwidth, dtype=int_dtype,
        device=device).view(dtype).to(hp_dtype).sort()[0]
    break_points = break_points[torch.isnan(break_points) == False]

    return break_points


BREAK_POINTS = {t: calculate_break_points(t) for t in BITS.keys()}

INTERVALS = {
    t: (BREAK_POINTS[t][1:] - BREAK_POINTS[t][:-1]).to(hp_dtype)
    for t in BITS.keys()
}


def dge_fwd(x, dtype):
    break_points = BREAK_POINTS[dtype].to(x.device)
    intervals = INTERVALS[dtype].to(x.device)
    idx = torch.searchsorted(break_points, x, right=False, out_int32=True) - 1
    idx = torch.where(idx < 0, 0, idx)
    idx = torch.where(idx >= len(break_points) - 1, len(break_points) - 2, idx)
    delta = intervals[idx]
    half_delta = delta / 2.
    mid_point = break_points[idx] + half_delta
    return half_delta**(1 - 1 / k) * torch.sign(x - mid_point) * (
        torch.abs(x - mid_point)**(1 / k)) + mid_point


def dge_bwd(x, dtype):
    break_points = BREAK_POINTS[dtype].to(x.device)
    intervals = INTERVALS[dtype].to(x.device)
    idx = torch.searchsorted(break_points, x, right=False, out_int32=True) - 1
    idx = torch.where(idx < 0, 0, idx)
    idx = torch.where(idx >= len(break_points) - 1, len(break_points) - 2, idx)
    delta = intervals[idx]
    half_delta = delta / 2.
    mid_point = break_points[idx] + half_delta
    return torch.clamp(
        half_delta**(1 - 1 / k) * (torch.abs(x - mid_point)**(1 / k - 1)) / k,
        0, 3.)
