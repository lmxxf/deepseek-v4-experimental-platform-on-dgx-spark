"""
离线分析探针激活值：对比 4 路 HC 的三种折算方式在「诗=正交、白话=冗余」上的信号。
纯 numpy/torch CPU，宿主机直接跑，不需要容器。
"""
import torch
import numpy as np

D = torch.load("probe_poem_acts.pt", map_location="cpu", weights_only=False)
LAYERS = D["target_layers"]
print(f"target_layers={LAYERS}, hc_mult={D['hc_mult']}, dim={D['dim']}\n")


def fold(act, mode):
    """act: [1, seq, 4, 4096] -> [seq, d]"""
    a = act[0].float()  # [seq, 4, 4096]
    if mode == "stream0":
        return a[:, 0, :].numpy()
    if mode == "sum":
        return a.sum(dim=1).numpy()
    if mode == "concat":
        return a.reshape(a.shape[0], -1).numpy()
    raise ValueError(mode)


def cosine_high_pairs(h, thresh=0.8, min_dist=2):
    norms = np.linalg.norm(h, axis=-1, keepdims=True)
    hn = h / (norms + 1e-10)
    cos = hn @ hn.T
    n = h.shape[0]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) > min_dist and cos[i][j] > thresh:
                count += 1
    return count


def eid(h):
    _, S, _ = np.linalg.svd(h, full_matrices=False)
    s2 = S ** 2
    s2 = s2[s2 > 1e-12]
    p = s2 / s2.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


results = D["results"]
# 按 pair_id 分组
pair_ids = sorted(set(k.rsplit("_", 1)[0] for k in results))

for pid in pair_ids:
    poem = results[f"{pid}_poem"]
    plain = results[f"{pid}_plain"]
    print(f"=== {poem['type']} ({poem['source']}) ===")
    print(f"  诗  : {poem['text']}  ({len(poem['tokens'])} tokens)")
    print(f"  白话: {plain['text']}  ({len(plain['tokens'])} tokens)\n")
    for lid in LAYERS:
        print(f"  --- Layer {lid} ---")
        print(f"  {'折算':<10}{'诗·高相似对':<14}{'白话·高相似对':<16}"
              f"{'诗·归一EID':<14}{'白话·归一EID':<16}{'EID比值':<10}")
        for mode in ("stream0", "sum", "concat"):
            ph = fold(poem["activations"][lid], mode)
            plh = fold(plain["activations"][lid], mode)
            p_hi = cosine_high_pairs(ph)
            pl_hi = cosine_high_pairs(plh)
            p_eid = eid(ph) / len(poem["tokens"])
            pl_eid = eid(plh) / len(plain["tokens"])
            ratio = p_eid / pl_eid if pl_eid > 0 else float("inf")
            print(f"  {mode:<10}{p_hi:<14}{pl_hi:<16}{p_eid:<14.3f}{pl_eid:<16.3f}{ratio:<10.2f}")
        print()
    print()
