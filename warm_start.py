import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- config -----
EMB_DIM = 16
HIDDEN = [64, 32]
DROPOUT = 0.2

# Example cardinalities after LabelEncoding (fit on train)
card = {
    "user_id": 120_000,
    "region": 200,
    "device": 5,
    "gender": 3,
    "banner_id": 50_000,     # grows over time
    "brand": 8_000,
    "vertical": 200,
    "language": 30,
    "price_tier": 6,
}

numeric_feats = ["age", "tenure_days", "discount_pct"]  # example

class DeepFM(nn.Module):
    def __init__(self, card, emb_dim=EMB_DIM, hidden=HIDDEN, p=DROPOUT, num_numeric=3):
        super().__init__()

        # Embeddings per categorical "field"
        self.emb = nn.ModuleDict({
            k: nn.Embedding(num_embeddings=v, embedding_dim=emb_dim)
            for k, v in card.items()
        })

        # First-order (linear) terms for each field (treat each field as a single index feature)
        self.lin = nn.ModuleDict({
            k: nn.Embedding(num_embeddings=v, embedding_dim=1)
            for k, v in card.items()
        })
        self.lin_num = nn.Linear(num_numeric, 1)

        # DNN on concatenated embeddings + numerics
        dnn_in = emb_dim * len(card) + num_numeric
        layers = []
        last = dnn_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p)]
            last = h
        self.dnn = nn.Sequential(*layers)
        self.out = nn.Linear(last + 1 + 1, 1)  # concat(FM bi-term, linear, dnn) → logit

        # Metadata->banner embedding synthesizer (for warm start)
        # Input = concat(brand_emb, vertical_emb, language_emb, price_tier_emb)
        meta_in = emb_dim * 4
        self.banner_meta2emb = nn.Sequential(
            nn.Linear(meta_in, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_num):
        """
        x_cat: dict of field -> LongTensor shape [B]
        x_num: FloatTensor shape [B, num_numeric]
        """
        # embeddings per field
        E = {k: self.emb[k](x_cat[k]) for k in self.emb.keys()}             # [B, d]
        L = [self.lin[k](x_cat[k]) for k in self.lin.keys()]                # list of [B, 1]
        L_num = self.lin_num(x_num)                                         # [B, 1]

        # ----- FM bi-interaction term: 0.5 * ( (sum e)^2 - sum(e^2) ) -----
        stackE = torch.stack(list(E.values()), dim=1)   # [B, F, d]
        sumE = stackE.sum(dim=1)                        # [B, d]
        sumE_sq = sumE * sumE                           # [B, d]
        sqE = stackE * stackE                           # [B, F, d]
        sqE_sum = sqE.sum(dim=1)                        # [B, d]
        fm_bi = 0.5 * (sumE_sq - sqE_sum)               # [B, d]
        fm_bi_term = fm_bi.sum(dim=1, keepdim=True)     # [B, 1]

        # ----- linear term (first-order) -----
        lin_term = torch.stack(L, dim=1).sum(dim=1) + L_num  # [B, 1]

        # ----- deep term -----
        concatE = torch.cat(list(E.values()), dim=1)   # [B, F*d]
        dnn_in = torch.cat([concatE, x_num], dim=1)    # [B, F*d + num_numeric]
        dnn_out = self.dnn(dnn_in)                     # [B, H]
        logit = self.out(torch.cat([fm_bi_term, lin_term, dnn_out], dim=1))
        prob = torch.sigmoid(logit)
        return prob

    @torch.no_grad()
    def warm_start_banner_embedding(self, new_banner_idx, brand_idx, vertical_idx, language_idx, price_tier_idx, alpha=1.0):
        """
        Create/refresh the embedding for a NEW banner_id row using metadata.
        alpha=1.0 → pure meta init, <1.0 blends with current (random) row.
        """
        brand_emb = self.emb["brand"](torch.tensor([brand_idx], dtype=torch.long, device=self.emb["brand"].weight.device))
        vert_emb  = self.emb["vertical"](torch.tensor([vertical_idx], dtype=torch.long, device=self.emb["vertical"].weight.device))
        lang_emb  = self.emb["language"](torch.tensor([language_idx], dtype=torch.long, device=self.emb["language"].weight.device))
        tier_emb  = self.emb["price_tier"](torch.tensor([price_tier_idx], dtype=torch.long, device=self.emb["price_tier"].weight.device))

        meta_vec = torch.cat([brand_emb, vert_emb, lang_emb, tier_emb], dim=1)  # [1, 4d]
        pred_banner_vec = self.banner_meta2emb(meta_vec)                        # [1, d]

        W = self.emb["banner_id"].weight
        if new_banner_idx >= W.shape[0]:
            # expand table if needed (optional: preallocate slack rows in practice)
            pad = new_banner_idx - W.shape[0] + 1
            new_rows = torch.zeros(pad, W.shape[1], device=W.device)
            nn.init.normal_(new_rows, mean=0.0, std=0.01)
            self.emb["banner_id"].weight = nn.Parameter(torch.cat([W, new_rows], dim=0))

        # blend initialize
        cur = self.emb["banner_id"].weight[new_banner_idx].clone()
        self.emb["banner_id"].weight[new_banner_idx] = alpha * pred_banner_vec.squeeze(0) + (1 - alpha) * cur
