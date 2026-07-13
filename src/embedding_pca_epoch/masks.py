import torch

class MaskingStrategy:
    def __init__(self, strategy="low_pt", min_context=15, num_target=5):
        self.strategy = strategy
        self.min_context = min_context
        self.num_target = num_target

    def set_strategy(self, strategy: str):
        self.strategy = strategy

    def __call__(self, batch_data, device):
        batch_size, seq_len, num_features = batch_data.shape
        batch_data = batch_data.to(device)

        masks_context = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        masks_target = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

        pT = batch_data[:, :, 0]
        valid_particles = pT > 0

        for i in range(batch_size):
            valid_indices = valid_particles[i].nonzero().flatten()
            if valid_indices.numel() == 0:
                continue

            if self.strategy in ["low_pt", "high_pt", "random"]:
                pT_i = pT[i, valid_indices]
                num_valid = valid_indices.shape[0]

                if self.strategy == "low_pt":
                    sorted = valid_indices[pT_i.argsort(descending=False)]
                elif self.strategy == "high_pt":
                    sorted = valid_indices[pT_i.argsort(descending=True)]
                elif self.strategy == "random":
                    sorted = valid_indices[torch.randperm(num_valid)]

            elif self.strategy == "high_variance":
                features = batch_data[i, valid_indices]
                mean = features.mean(dim=0, keepdim=True)
                deviation = (features - mean).pow(2).mean(dim=1)  # variance score
                sorted = valid_indices[deviation.argsort(descending=True)]

            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            target_indices = sorted[:self.num_target]
            remaining_indices = sorted[self.num_target:]
            num_context = min(len(remaining_indices), self.min_context)
            context_indices = remaining_indices[:num_context]

            masks_context[i, context_indices] = 1.0
            masks_target[i, target_indices] = 1.0

        return masks_context, masks_target
