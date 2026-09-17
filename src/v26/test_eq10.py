import torch, torch.nn.functional as F
torch.manual_seed(0); B, K, eps = 64, 7, 0.05
logits = torch.randn(B, K, dtype=torch.float64); y = torch.randint(0, K, (B,))
w = torch.rand(K, dtype=torch.float64) * 10 + 0.1
ref = F.cross_entropy(logits, y, weight=w, label_smoothing=eps)          # reduction='mean'
q = torch.full((B, K), eps / K, dtype=torch.float64); q[torch.arange(B), y] += 1 - eps
logp = F.log_softmax(logits, dim=1)
eq10 = -(w[None, :] * q * logp).sum() / w[y].sum()                      # Eq. (10) of the paper
alt = -(w[y][:, None] * q * logp).sum() / w[y].sum()                    # w_{y_b} outside the inner sum
assert torch.allclose(ref, eq10, rtol=0, atol=1e-12), (ref, eq10)
assert not torch.allclose(ref, alt), 'the alternative placement must differ'
print('Eq.(10) == PyTorch:', float(ref), float(eq10), '| alternative:', float(alt))
