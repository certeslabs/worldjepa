# benchmarks/next_frame_retrieval.py
import torch
import torch.nn.functional as F


@torch.no_grad()
def next_frame_retrieval_hard(Z_pred, Z, k=1, min_gap=4):
    """
    Retrieval avec hard negatives intra-vidéo.

    Arguments:
        Z_pred: (B, T, D) — prédictions du predictor
        Z:      (B, T, D) — embeddings encodeur
        k:      Recall@k
        min_gap: distance minimale temporelle pour les négatifs

    Returns:
        Recall@k — float

    Note: Les négatifs sont des frames du MÊME clip mais éloignées de
    min_gap positions. Cela force le modèle à modéliser la dynamique
    temporelle fine, pas juste l'identité vidéo.
    """
    B, T, D = Z.shape
    correct = 0
    total = 0

    Z_pred_norm = F.normalize(Z_pred, dim=-1)
    Z_norm = F.normalize(Z, dim=-1)

    for b in range(B):
        for t in range(T - 1):
            q = Z_pred_norm[b, t]  # query: prédiction de t+1
            pos = Z_norm[b, t + 1]  # positif: vraie frame t+1

            # Négatifs: même vidéo, loin dans le temps
            neg_indices = [i for i in range(T) if abs(i - (t + 1)) > min_gap]
            if len(neg_indices) == 0:
                continue

            negs = Z_norm[b, neg_indices]  # (N_neg, D)

            # Similarités: [positif, neg1, neg2, ...]
            sims = torch.cat(
                [(q * pos).sum().unsqueeze(0), (q.unsqueeze(0) * negs).sum(dim=1)]
            )

            rank = (sims >= sims[0]).sum().item()
            if rank <= k:
                correct += 1
            total += 1

    return correct / max(total, 1)


@torch.no_grad()
def next_frame_retrieval_global(Z_pred, Z, k=1):
    """
    Retrieval avec distracteurs aléatoires globaux (sanity check).

    À utiliser en COMPLÉMENT du hard retrieval, pas à sa place.
    Un R@1 global élevé sans hard R@1 élevé = faux positif.
    """
    B, T, D = Z.shape

    # Flatten: toutes les paires (b, t) → (b, t+1)
    preds = Z_pred[:, :-1].reshape(-1, D)
    targets = Z[:, 1:].reshape(-1, D)

    preds = F.normalize(preds, dim=-1)
    targets = F.normalize(targets, dim=-1)

    N = preds.shape[0]
    sims = preds @ targets.T  # (N, N)
    ranks = (sims >= sims.diag().unsqueeze(1)).sum(dim=1)
    return (ranks <= k).float().mean().item()


@torch.no_grad()
def run_retrieval_suite(Z_pred, Z, k_values=(1, 5, 10)):
    """
    Lance les deux protocoles et retourne un dict complet.

    Utilisation dans train.py :
        retrieval = run_retrieval_suite(out["Z_pred"], out["Z"])
        log.update(retrieval)
    """
    results = {}
    for k in k_values:
        results[f"R@{k}_global"] = next_frame_retrieval_global(Z_pred, Z, k=k)
        results[f"R@{k}_hard"] = next_frame_retrieval_hard(Z_pred, Z, k=k)
    return results
