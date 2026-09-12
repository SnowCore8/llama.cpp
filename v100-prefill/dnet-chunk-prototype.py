#!/usr/bin/env python3
# delta-net 分块（chunked）算法的数值可行性原型。
# 目的：写 CUDA 之前，用门禁同分布的输入 + 门禁同判据（NMSE）验证分块公式是否安全。
# 参考实现 = CPU/CUDA 现行串行 recurrence（非 KDA，标量衰减）。
# 分块实现 = 本脚本推导的形式：三角求解与状态、衰减解耦（ū = u/Γ）。
import numpy as np

rng = np.random.default_rng(20260912)


def l2_norm(x, eps=1e-6):
    # ggml_l2_norm: x / max(||x||_2, eps)，沿 dim 0
    n = np.sqrt(np.maximum((x * x).sum(axis=0, keepdims=True), 0.0))
    return x / np.maximum(n, eps)


def make_inputs(n_tokens, d, seed):
    r = np.random.default_rng(seed)
    q = l2_norm(r.uniform(-1, 1, (n_tokens, d)))
    k = l2_norm(r.uniform(-1, 1, (n_tokens, d)))
    v = r.uniform(-0.3, 5.0, (n_tokens, d))
    g = r.uniform(-20.0, -1e-4, n_tokens)
    beta = r.uniform(0.0, 1.0, n_tokens)
    S = r.uniform(-1, 1, (d, d))
    return q, k, v, g, beta, S


def ref_seq(q, k, v, g, beta, S):
    # 与 CPU/CUDA 现行实现同序：S ← γS; kv = Sᵀk; delta = (v−kv)β; S += k⊗delta; o = (Sᵀq)·scale
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((n, d))
    S = S.astype(np.float64).copy()
    for t in range(n):
        gm = np.exp(float(g[t]))
        S *= gm
        kv = S.T @ k[t]
        delta = (v[t] - kv) * beta[t]
        S += np.outer(k[t], delta)
        out[t] = (S.T @ q[t]) * scale
    return out, S


def chunked(q, k, v, g, beta, S, Cc):
    # 安全形式：不做 u/Γ 归一化（Γ 下溢会炸），所有衰减权重在 log 空间取 exp(差)，恒在 (0,1]。
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((n, d))
    S = S.astype(np.float64).copy()
    for c0 in range(0, n, Cc):
        c1 = min(c0 + Cc, n)
        m = c1 - c0
        K = k[c0:c1].astype(np.float64)
        Q = q[c0:c1].astype(np.float64)
        V = v[c0:c1].astype(np.float64)
        gC = g[c0:c1].astype(np.float64)
        B = beta[c0:c1].astype(np.float64)

        GS = np.cumsum(gC)                            # log 空间累计衰减
        tril = np.tril(np.ones((m, m), bool))
        D = GS[:, None] - GS[None, :]                 # ≤ 0（下三角），上三角不用
        R = np.exp(np.where(tril, D, -np.inf))        # R[t][s] = Γ_t/Γ_s ∈ (0,1]

        KK = K @ K.T
        A = np.tril(KK, -1) * R * B[:, None]          # I + A 与状态无关，只含衰减权重
        T = np.linalg.inv(np.eye(m) + A)
        P = T @ (B[:, None] * V)                      # T·β·v
        Gam = np.exp(GS)
        W = T * (B * Gam)[None, :]                    # T·diag(β·Γ)
        u = P - W @ (K @ S)                           # u = Tβ(v − Γ⊙Sᵀk)

        QK = Q @ K.T
        out[c0:c1] = (Gam[:, None] * (Q @ S) + (np.tril(QK) * R) @ u) * scale
        S = Gam[-1] * S + (K * R[m - 1, :][:, None]).T @ u
    return out, S


def nmse(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(((a - b) ** 2).sum() / max((a * a).sum(), 1e-300))


def run_case(n_tokens, d, Cc, seed):
    q, k, v, g, beta, S = make_inputs(n_tokens, d, seed)
    o_ref, s_ref = ref_seq(q, k, v, g, beta, S)
    o_chk, s_chk = chunked(q, k, v, g, beta, S, Cc)
    return nmse(o_ref, o_chk), nmse(s_ref, s_chk)


if __name__ == "__main__":
    print("门禁判据 NMSE = Σ(a-b)²/Σa²，阈值 1e-7（RMS 相对 ~3.2e-4）")
    print("")
    hdr = f"{'d':>4} {'C':>5} {'n_tok':>6} {'Ck':>4} {'NMSE_attn':>12} {'NMSE_state':>12}"
    print(hdr)
    for d, n_tokens, Cc in [
        (64, 256, 16), (64, 256, 32), (64, 256, 64), (64, 256, 128),
        (128, 256, 32), (128, 256, 64), (128, 256, 128),
        (128, 4096, 64), (128, 4096, 128),
        (128, 100, 64), (128, 127, 64), (128, 65, 64), (128, 33, 64), (128, 2, 16),
    ]:
        ea, es = run_case(n_tokens, d, Cc, seed=hash((d, n_tokens)) & 0xffff)
        flag = "OK" if max(ea, es) < 1e-7 else ("WARN" if max(ea, es) < 1e-6 else "FAIL")
        print(f"{d:>4} {'-':>5} {n_tokens:>6} {Cc:>4} {ea:>12.3e} {es:>12.3e}  {flag}")

    print("")
    print("=== 极端衰减压力：g 全部取区间端点 ===")
    for d, n_tokens, Cc in [(128, 256, 64), (128, 256, 128)]:
        r = np.random.default_rng(7)
        q = l2_norm(r.uniform(-1, 1, (n_tokens, d)))
        k = l2_norm(r.uniform(-1, 1, (n_tokens, d)))
        v = r.uniform(-0.3, 5.0, (n_tokens, d))
        S = r.uniform(-1, 1, (d, d))
        b = r.uniform(0, 1, n_tokens)
        for gname, g in [("g=-1e-4 (几乎不衰减)", np.full(n_tokens, -1e-4)),
                         ("g=-20 (强衰减)", np.full(n_tokens, -20.0)),
                         ("g=-20..-1e-4 交替", np.where(np.arange(n_tokens) % 2 == 0, -1e-4, -20.0)),
                         ("β≡1", r.uniform(0, 1, n_tokens) * 0 + 1.0)]:
            o_ref, s_ref = ref_seq(q, k, v, g, b, S)
            o_chk, s_chk = chunked(q, k, v, g, b, S, Cc)
            print(f"d={d} n={n_tokens} C={Cc} {gname:>24}: attn {nmse(o_ref,o_chk):.3e} state {nmse(s_ref,s_chk):.3e}")
