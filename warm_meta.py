class FactorizationMachine(nn.Module):
    """
    Factorization Machine with warm-start from banner metadata.
    - categorical_field_dims: list of cardinalities for each categorical field (same order as categorical_field_names)
    - categorical_field_names: list of field names in the SAME order as categorical cols in the dataset
    - banner_field_name: name of the banner id field (default 'banner_id')
    - banner_meta_fields: list of categorical field names that describe banner metadata (used to synthesize a warm-start embedding)
    """
    def __init__(
        self,
        n_numeric_features,
        categorical_field_dims,
        embed_dim,
        banner_field_idx,  # kept for backward-compat, will be overwritten if names provided
        dropout_rate=0.1,
        categorical_field_names=None,           # NEW
        banner_field_name: str = "banner_id",   # NEW
        banner_meta_fields: list[str] = None    # NEW
    ):
        super().__init__()
        self.n_numeric_features = n_numeric_features
        self.categorical_field_dims = categorical_field_dims
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.bias = nn.Parameter(torch.zeros((1,)))

        # ---- Names & indices ----
        self.cat_names = categorical_field_names or [f"cat_{i}" for i in range(len(categorical_field_dims))]
        self.field_index = {n:i for i,n in enumerate(self.cat_names)}
        # prefer name lookup when available
        self.banner_field_idx = self.field_index.get(banner_field_name, banner_field_idx)
        self.banner_meta_fields = banner_meta_fields or []

        # --- Linear Part (w_i * x_i) ---
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, 1) for num_embeddings in categorical_field_dims
        ])
        if self.n_numeric_features > 0:
            self.linear_numeric = nn.Linear(self.n_numeric_features, 1)

        # --- Interaction Part (v_i, v_j) ---
        self.interaction_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embed_dim) for num_embeddings in categorical_field_dims
        ])
        if self.n_numeric_features > 0:
            self.interaction_numeric_vectors = nn.Parameter(torch.randn(n_numeric_features, embed_dim) * 0.01)

        # --- Meta -> banner embedding synthesizers (for warm start) ---
        if len(self.banner_meta_fields) > 0:
            meta_in = embed_dim * len(self.banner_meta_fields)
            self.meta2emb = nn.Sequential(
                nn.Linear(meta_in, 2*embed_dim), nn.ReLU(),
                nn.Linear(2*embed_dim, embed_dim)
            )
            # (optional) meta → linear bias for banner
            self.meta2lin = nn.Sequential(
                nn.Linear(meta_in, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )
        else:
            self.meta2emb, self.meta2lin = None, None

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_numeric, x_categorical):
        # ----- linear -----
        linear_terms = self.bias
        cat_linear_terms = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        linear_terms = linear_terms + torch.sum(torch.cat(cat_linear_terms, dim=1), dim=1, keepdim=True)
        if self.n_numeric_features > 0:
            linear_terms = linear_terms + self.linear_numeric(x_numeric)

        # ----- interaction (FM bi-term) -----
        cat_interaction_vectors = [emb(x_categorical[:, i]) for i, emb in enumerate(self.interaction_embeddings)]
        if self.n_numeric_features > 0:
            numeric_interaction_vectors = x_numeric.unsqueeze(2) * self.interaction_numeric_vectors.unsqueeze(0)
            all_vectors = torch.cat(cat_interaction_vectors + [numeric_interaction_vectors], dim=1)
        else:
            all_vectors = torch.cat(cat_interaction_vectors, dim=1)

        all_vectors = self.dropout(all_vectors)
        sum_of_squares = torch.sum(all_vectors, dim=1).pow(2)
        square_of_sums = torch.sum(all_vectors.pow(2), dim=1)
        interaction_terms = 0.5 * torch.sum(sum_of_squares - square_of_sums, dim=1, keepdim=True)

        logits = linear_terms + interaction_terms
        return logits.squeeze(1)

    @torch.no_grad()
    def handle_new_banners(
        self,
        new_banner_id: int,
        strategy: str = 'cold',
        source_banner_id: int = None,
        source_group_ids: list[int] = None,
        meta_values: dict[str, int] = None,   # NEW: {field_name: label_id}
        alpha: float = 0.9                     # NEW: blend factor for warm_meta
    ):
        """
        Warm-start strategies:
        - 'cold': random small init (existing behavior)
        - 'warm_copy': copy from source banner id (existing)
        - 'warm_average': average a group of banners (existing)
        - 'warm_meta': synthesize from metadata embeddings (NEW)
        """
        bidx = self.banner_field_idx

        if strategy == 'cold':
            nn.init.normal_(self.embeddings[bidx].weight[new_banner_id], 0, 0.01)
            nn.init.normal_(self.interaction_embeddings[bidx].weight[new_banner_id], 0, 0.01)
            print(f"Cold start for new banner ID: {new_banner_id}")
            return

        if strategy == 'warm_copy' and source_banner_id is not None:
            self.embeddings[bidx].weight[new_banner_id] = self.embeddings[bidx].weight[source_banner_id].clone()
            self.interaction_embeddings[bidx].weight[new_banner_id] = self.interaction_embeddings[bidx].weight[source_banner_id].clone()
            print(f"Warm start for banner {new_banner_id}, copying from {source_banner_id}")
            return

        if strategy == 'warm_average' and source_group_ids is not None:
            avg_linear_weight = self.embeddings[bidx].weight[source_group_ids].mean(dim=0)
            avg_interaction_weight = self.interaction_embeddings[bidx].weight[source_group_ids].mean(dim=0)
            self.embeddings[bidx].weight[new_banner_id] = avg_linear_weight
            self.interaction_embeddings[bidx].weight[new_banner_id] = avg_interaction_weight
            print(f"Warm start for banner {new_banner_id}, averaging from {len(source_group_ids)} banners.")
            return

        if strategy == 'warm_meta':
            if (self.meta2emb is None) or (meta_values is None) or (len(self.banner_meta_fields) == 0):
                raise ValueError("warm_meta requires banner_meta_fields and meta_values.")

            # collect metadata embeddings (interaction embeddings) and synthesize
            embs = []
            device = self.interaction_embeddings[0].weight.device
            for fname in self.banner_meta_fields:
                idx = self.field_index[fname]
                val = int(meta_values[fname])  # label-encoded id
                e = self.interaction_embeddings[idx](torch.tensor([val], dtype=torch.long, device=device))  # [1, d]
                embs.append(e)
            meta_cat = torch.cat(embs, dim=1)  # [1, len(meta_fields)*d]
            pred_vec = self.meta2emb(meta_cat)  # [1, d]
            pred_lin = self.meta2lin(meta_cat) if self.meta2lin is not None else torch.zeros(1,1, device=device)

            # blend with current (random) row to keep variance reasonable
            cur_vec = self.interaction_embeddings[bidx].weight[new_banner_id].clone()
            cur_lin = self.embeddings[bidx].weight[new_banner_id].clone()

            self.interaction_embeddings[bidx].weight[new_banner_id] = alpha * pred_vec.squeeze(0) + (1 - alpha) * cur_vec
            self.embeddings[bidx].weight[new_banner_id] = alpha * pred_lin.squeeze(0) + (1 - alpha) * cur_lin
            print(f"Warm start (metadata) for banner {new_banner_id} with alpha={alpha:.2f}.")
            return

        raise ValueError("Invalid strategy or missing arguments.")
