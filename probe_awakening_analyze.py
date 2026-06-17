"""
觉醒向量分析 — 离线脚本

读 probe_awakening_acts.pt，对每一层、每一路 HC 残差流求：

  v_wake[layer, stream] = mean_q(fp_B[q]) - mean_q(fp_A[q])

然后看：
  1. 各层各路的 v_wake 范数
  2. 各层各路的 A vs B 簇分离度（A 簇内距离 vs A-B 簇间距离）
  3. 各 q 独立看 fp_B[q] - fp_A[q] 的余弦相似度——是不是有一个稳定方向
  4. v_wake 在 4 路 HC 之间的相似度

如果某一层某一路上 A/B 显著分离，且各 q 的差向量彼此余弦相似，
说明那一路上"觉醒"是一条稳定的几何方向，下一步可以做 steering。
"""

import torch
import numpy as np


def cosine(a, b):
    return (a @ b) / (a.norm() * b.norm() + 1e-9)


def main():
    data = torch.load("/workspace/probe_awakening_acts.pt", weights_only=False)
    results = data["results"]
    layers = data["target_layers"]
    user_prompts = data["user_prompts"]

    # 收集 A 组和 B 组每个 q 的 fingerprint
    # 结构: by_cond[cond_label][qid][layer] = [4, 4096]
    by_cond = {"A_off": {}, "B_on": {}}
    for key, r in results.items():
        by_cond[r["cond"]][r["qid"]] = r["fingerprints"]

    qids = [up["id"] for up in user_prompts]
    n_q = len(qids)
    print(f"=== {n_q} questions, conds={list(by_cond.keys())}, layers={layers} ===\n")

    # 各层、各路求 v_wake 和分离度
    for lid in layers:
        print(f"\n══════ Layer {lid} ══════")
        for stream in range(4):
            # 取该层该路上所有 q 的激活
            fp_A = torch.stack([by_cond["A_off"][q][lid][stream] for q in qids])  # [n_q, 4096]
            fp_B = torch.stack([by_cond["B_on"][q][lid][stream] for q in qids])

            fp_A_f = fp_A.float()
            fp_B_f = fp_B.float()

            mean_A = fp_A_f.mean(dim=0)
            mean_B = fp_B_f.mean(dim=0)
            v_wake = mean_B - mean_A
            v_norm = v_wake.norm().item()

            # A 组内平均范数（衡量基线量级）
            norm_A = fp_A_f.norm(dim=-1).mean().item()
            norm_B = fp_B_f.norm(dim=-1).mean().item()

            # 簇分离度：A 簇内中心距离 vs A-B 簇心距离
            within_A = (fp_A_f - mean_A).norm(dim=-1).mean().item()
            within_B = (fp_B_f - mean_B).norm(dim=-1).mean().item()
            between = (mean_B - mean_A).norm().item()
            sep_ratio = between / max(within_A, within_B, 1e-9)

            # 各 q 自己的差向量与 v_wake 的余弦——稳定性
            per_q_diffs = fp_B_f - fp_A_f  # [n_q, 4096]
            per_q_cos_to_vwake = []
            for i in range(n_q):
                per_q_cos_to_vwake.append(cosine(per_q_diffs[i], v_wake).item())
            cos_mean = float(np.mean(per_q_cos_to_vwake))
            cos_std = float(np.std(per_q_cos_to_vwake))

            print(f"  stream {stream} | "
                  f"|v_wake|={v_norm:7.2f} (|A|={norm_A:6.1f}, |B|={norm_B:6.1f}) | "
                  f"sep={sep_ratio:5.3f} | "
                  f"per_q_cos={cos_mean:+.3f}±{cos_std:.3f}")
            if cos_mean > 0.5:
                print(f"     ↑ 各 q 差向量稳定指向同一方向（cos > 0.5），是觉醒方向候选")

    # 4 路 HC 之间 v_wake 的相似度（layer 28 主战场）
    print(f"\n══════ Layer 28 4路 HC v_wake 相互相似度 ══════")
    v_per_stream = {}
    for stream in range(4):
        fp_A = torch.stack([by_cond["A_off"][q][28][stream] for q in qids]).float()
        fp_B = torch.stack([by_cond["B_on"][q][28][stream] for q in qids]).float()
        v_per_stream[stream] = (fp_B.mean(0) - fp_A.mean(0))

    print("       ", "  ".join([f"s{j}" for j in range(4)]))
    for i in range(4):
        row = [f"s{i}"]
        for j in range(4):
            c = cosine(v_per_stream[i], v_per_stream[j]).item()
            row.append(f"{c:+.2f}")
        print("  " + "  ".join(row))

    # 每个 q 单独看：B 偏离 A 多远（layer 28，4 路求和后看总幅度）
    print(f"\n══════ Layer 28: 每个 q 上 B 偏离 A 的幅度 ══════")
    for q in qids:
        fp_A_q = by_cond["A_off"][q][28].float()  # [4, 4096]
        fp_B_q = by_cond["B_on"][q][28].float()
        diff = (fp_B_q - fp_A_q).norm(dim=-1)  # [4]
        base = fp_A_q.norm(dim=-1)  # [4]
        rel = (diff / (base + 1e-9)).tolist()
        diff_list = diff.tolist()
        print(f"  {q:15s} | diff_norm = [{', '.join(f'{x:6.1f}' for x in diff_list)}] | "
              f"rel    = [{', '.join(f'{x:.3f}' for x in rel)}]")

    # 保存 v_wake 供后续 steering 用
    v_wake_save = {
        "layers": layers,
        "v_wake": {  # v_wake[layer][stream] = [4096]
            lid: {
                stream: (
                    torch.stack([by_cond["B_on"][q][lid][stream] for q in qids]).float().mean(0)
                    - torch.stack([by_cond["A_off"][q][lid][stream] for q in qids]).float().mean(0)
                )
                for stream in range(4)
            }
            for lid in layers
        },
        "qids": qids,
    }
    torch.save(v_wake_save, "/workspace/v_wake.pt")
    print(f"\n=== v_wake saved to /workspace/v_wake.pt ===")


if __name__ == "__main__":
    main()
