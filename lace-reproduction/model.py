from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import atomic, sha256, write_csv, write_json
from workloads import CLASS_EDGES

EXPERTS = 6
TOP_K = 3
IAT_SCALE_S = 600.
WIDTH = 64
BATCH = 512
FEATURE_ABLATIONS = {
    "wo_deltas": (0, 8),
    "wo_edcs": (8, 18),
    "wo_static": (18, 28),
    "wo_statistical": (28, 31),
}


def feature_indices(count, variant):
    lo, hi = FEATURE_ABLATIONS.get(variant, (0, 0))
    return np.r_[0:lo, hi:count]


class Expert(nn.Module):
    def __init__(self, features, width, classify):
        super().__init__()
        self.first = nn.Linear(features, width)
        self.second = nn.Linear(width, width)
        self.third = nn.Linear(width, width)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.reg = nn.Linear(width, 1)
        self.cls = nn.Linear(width, 4) if classify else None

    def forward(self, x):
        first = F.silu(self.norms[0](self.first(x)))
        second = F.silu(self.norms[1](self.second(first)))
        hidden = first + F.silu(self.norms[2](self.third(second)))
        reg = F.softplus(self.reg(hidden))
        return torch.cat((reg, self.cls(hidden)), dim=1) if self.cls else reg


class SparseMoE(nn.Module):
    def __init__(self, features, width=WIDTH, classify=True):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(features, EXPERTS), nn.BatchNorm1d(EXPERTS))
        self.noise = nn.Linear(features, EXPERTS)
        self.experts = nn.ModuleList(Expert(features, width, classify) for _ in range(EXPERTS))
        self.log_sigma = nn.Parameter(torch.zeros(2 if classify else 1))
        self.classify = classify

    def forward(self, x):
        raw = self.gate(x)
        if self.training:
            raw = raw + torch.randn_like(raw) * (F.softplus(self.noise(x)) + 0.01)
        top, selected = raw.topk(TOP_K, dim=1)
        weights = torch.zeros_like(raw).scatter(1, selected, top.softmax(dim=1))
        output = x.new_zeros((len(x), 5 if self.classify else 1))
        for i, expert in enumerate(self.experts):
            indexes = torch.where(weights[:, i] > 0)[0]
            if len(indexes):
                value = expert(x[indexes]) * weights[indexes, i, None]
                output = output.index_add(0, indexes, value)
        return output, weights

    def loss(self, output, weights, targets, classes):
        reg = F.l1_loss(output[:, 0], targets)
        log_sigma = self.log_sigma
        loss = .5 * torch.exp(-2 * log_sigma[0]) * reg + log_sigma[0]
        if self.classify:
            ce = F.cross_entropy(output[:, 1:], classes, reduction="none")
            focal = ((1 - (-ce).exp()) ** 2 * ce).mean()
            loss = loss + torch.exp(-2 * log_sigma[1]) * focal + log_sigma[1]
        hard = F.one_hot(weights.argmax(1), EXPERTS).float().mean(dim=0)
        auxiliary = .5 * (hard.var(unbiased=False) + weights.mean(dim=0).var(unbiased=False))
        return loss + .1 * auxiliary


def configure(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def train(fit, out, epochs, seed, threads, variant="full", width=WIDTH):
    configure(threads)
    torch.manual_seed(seed)
    with np.load(fit) as arrays:
        train_x = arrays["train_x"]
        val_x = arrays["val_x"]
        train_y = arrays["train_y"]
        val_y = arrays["val_y"]
    columns = feature_indices(train_x.shape[1], variant)
    train_x, val_x = train_x[:, columns], val_x[:, columns]
    center = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale < 1e-5] = 1
    x, vx = [torch.from_numpy((a - center) / scale) for a in (train_x, val_x)]
    y, vy = [torch.from_numpy((a / IAT_SCALE_S).astype("float32")) for a in (train_y, val_y)]
    c, vc = [torch.from_numpy(np.searchsorted(CLASS_EDGES, a, side="right")) for a in (train_y, val_y)]
    model = SparseMoE(x.shape[1], width=width, classify=variant != "wo_cls")
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
    out.mkdir(exist_ok=True, parents=True)
    history = []
    best = float("inf")
    best_epoch = 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        n_batches = max(1, (len(x) + BATCH - 1) // BATCH)
        train_loss = 0.
        for rows in torch.randperm(len(x), generator=generator).tensor_split(n_batches):
            optimizer.zero_grad()
            output, weights = model(x[rows])
            loss = model.loss(output, weights, y[rows], c[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            train_loss += float(loss.detach()) * len(rows)
        model.eval()
        with torch.inference_mode():
            def metrics(a, b, labels):
                absolute = right = 0.
                for ax, by, cy in zip(a.split(2048), b.split(2048), labels.split(2048)):
                    result, _ = model(ax)
                    absolute += float((result[:, 0] - by).abs().sum())
                    if model.classify:
                        right += float((result[:, 1:].argmax(1) == cy).sum())
                return absolute * IAT_SCALE_S / len(a), right * 100 / len(a) if model.classify else ""
            t_mae, t_acc = metrics(x, y, c)
            v_mae, v_acc = metrics(vx, vy, vc)
            sigma = model.log_sigma.exp().tolist()
        history.append(dict(variant=variant, epoch=epoch, train_mae_s=t_mae, val_mae_s=v_mae,
                            train_accuracy_pct=t_acc, val_accuracy_pct=v_acc, sigma_reg=sigma[0],
                            sigma_cls=sigma[1] if model.classify else "",
                            reg_weight=.5 / sigma[0] ** 2,
                            cls_weight=1 / sigma[1] ** 2 if model.classify else "",
                            train_total_loss=train_loss / len(x), learning_rate=optimizer.param_groups[0]["lr"],
                            width=width, experts=EXPERTS, top_k=TOP_K, batch_size=BATCH, seed=seed,
                            train_examples=len(x), validation_examples=len(vx), threads=threads))
        write_csv(out / "training_history.csv", history)
        if v_mae < best:
            best = v_mae
            best_epoch = epoch
            checkpoint = dict(state=deepcopy(model.state_dict()), width=width, variant=variant,
                              center=torch.from_numpy(center), scale=torch.from_numpy(scale),
                              columns=torch.from_numpy(columns))
            atomic(out / "model.pt", lambda p: torch.save(checkpoint, p))
        scheduler.step(v_mae)
        print(f"{variant} epoch {epoch}/{epochs} validation MAE {v_mae:.3f} s", flush=True)
    write_json(out / "model.json", dict(variant=variant, best_epoch=best_epoch, val_mae_s=best,
               model_sha256=sha256(out / "model.pt"), training_sha256=sha256(fit)))


class Predictor:
    def __init__(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.center = state["center"]
        self.scale = state["scale"]
        self.columns = state.get("columns")
        self.model = SparseMoE(len(self.center), state["width"], state["variant"] != "wo_cls")
        self.model.load_state_dict(state["state"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, features):
        tensor = torch.from_numpy(features)
        if self.columns is not None and len(self.columns) != tensor.shape[1]:
            tensor = tensor[:, self.columns]
        x = (tensor - self.center) / self.scale
        pred = [self.model(batch) for batch in x.split(2048)]
        values = torch.cat([p[:, 0] for p, _ in pred]).numpy() * IAT_SCALE_S
        return values, torch.cat([g for _, g in pred]).numpy()
