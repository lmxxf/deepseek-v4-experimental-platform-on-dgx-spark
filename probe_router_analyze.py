"""
离线分析 MoE 路由：诗里每个字点亮哪些专家 + 字之间专家重叠度。
纯 CPU，宿主机直接跑。
"""
import torch

D = torch.load("probe_router.pt", map_location="cpu", weights_only=False)
N_LAYERS = D["n_layers"]
N_EXP = D["n_routed_experts"]
TOPK = D["topk"]
N_HASH = D["n_hash_layers"]
print(f"{N_LAYERS} 层, {N_EXP} 专家选 {TOPK}, 前 {N_HASH} 层哈希路由（跟语义无关）\n")

results = D["results"]
pair_ids = sorted(set(k.rsplit("_", 1)[0] for k in results))

# 看几个代表层：1 个哈希层 + 早/中/晚分数层
SHOW_LAYERS = [0, N_HASH, N_LAYERS // 2, N_LAYERS - 1]
SHOW_LAYERS = sorted(set(l for l in SHOW_LAYERS if 0 <= l < N_LAYERS))


def jaccard(a, b):
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


for pid in pair_ids:
    poem = results[f"{pid}_poem"]
    print(f"=== {poem['type']} ({poem['source']}) ===")
    print(f"诗: {poem['text']}")
    toks = poem["tokens"]
    routing = poem["routing"]

    for lid in SHOW_LAYERS:
        tag = "哈希(语义无关)" if lid < N_HASH else "分数路由"
        print(f"\n--- Layer {lid} [{tag}] — 每个字的 top-{TOPK} 专家 ---")
        idx = routing[lid]  # [n_tokens, topk]
        for t, tok in enumerate(toks):
            experts = idx[t].tolist()
            print(f"  「{tok}」-> {sorted(experts)}")

    # 关键字之间专家重叠（用中后段分数层）
    probe_layer = N_LAYERS - 1
    idx = routing[probe_layer]
    print(f"\n--- Layer {probe_layer} 字两两专家重叠 (Jaccard, 0=完全分散 1=完全相同) ---")
    n = len(toks)
    # 只看非标点的实义字
    real = [(t, tok) for t, tok in enumerate(toks) if tok.strip() and tok not in ("，", "。", "、")]
    header = "       " + "".join(f"{tok:>5}" for _, tok in real)
    print(header)
    for ti, toki in real:
        row = f"  {toki:>4} "
        for tj, tokj in real:
            j = jaccard(idx[ti].tolist(), idx[tj].tolist())
            row += f"{j:>5.2f}"
        print(row)

    # 整首诗的专家多样性：用了多少个不同专家
    print(f"\n--- 专家使用多样性 (Layer {probe_layer}) ---")
    all_exp = idx.flatten().tolist()
    uniq = len(set(all_exp))
    print(f"  {n} 个字 × {TOPK} = {n*TOPK} 次激活，用了 {uniq} 个不同专家"
          f"（满分散={min(n*TOPK, N_EXP)}，全重叠={TOPK}）")
    print()
