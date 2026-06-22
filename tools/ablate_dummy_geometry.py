import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_jit import JiT, JiTHRM, create_jit_model


class TinyDenoiser(nn.Module):
    def __init__(
        self,
        net,
        class_num,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=5e-2,
        noise_scale=1.0,
        deep_segments=1,
    ):
        super().__init__()
        self.net = net
        self.class_num = class_num
        self.label_drop_prob = label_drop_prob
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        self.deep_segments = deep_segments

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.class_num), labels)

    def sample_t(self, n, device):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        if self.deep_segments > 1:
            if not hasattr(self.net, "forward_segments"):
                raise ValueError("deep_segments requires a model with forward_segments")
            x_preds = self.net.forward_segments(z, t.flatten(), labels_dropped, segments=self.deep_segments)
        else:
            x_preds = [self.net(z, t.flatten(), labels_dropped)]

        losses = []
        for x_pred in x_preds:
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
            losses.append(((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean())
        return torch.stack(losses).mean()


def _draw_geometry_from_labels(labels, sample_ids, image_size, device, class_num):
    batch_size = labels.numel()
    images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(image_size, device=device),
        torch.arange(image_size, device=device),
        indexing="ij",
    )
    xx = xx.float()
    yy = yy.float()

    for i in range(batch_size):
        label = int(labels[i].item())
        sample_id = int(sample_ids[i].item())
        rng = random.Random(label * 1000003 + sample_id * 9176 + image_size)

        palette = torch.tensor([
            ((label * 37) % 255) / 255.0,
            ((label * 67 + 51) % 255) / 255.0,
            ((label * 97 + 103) % 255) / 255.0,
        ], device=device)
        color = 0.35 + 0.65 * palette

        cx_base = image_size * (0.25 + 0.5 * (((label * 13) % max(class_num, 1)) / max(class_num - 1, 1)))
        cy_base = image_size * (0.25 + 0.5 * (((label * 29 + 7) % max(class_num, 1)) / max(class_num - 1, 1)))
        jitter = image_size // 10
        cx = int(max(image_size * 0.15, min(image_size * 0.85, cx_base + rng.randint(-jitter, jitter))))
        cy = int(max(image_size * 0.15, min(image_size * 0.85, cy_base + rng.randint(-jitter, jitter))))
        size = rng.randint(image_size // 8, image_size // 3)
        shape_id = label % 6

        if shape_id == 0:
            mask = (xx - cx).abs().maximum((yy - cy).abs()) <= size
        elif shape_id == 1:
            mask = (xx - cx).square() + (yy - cy).square() <= size * size
        elif shape_id == 2:
            mask = ((yy - cy).abs() <= size) & ((xx - cx).abs() <= (size - (yy - cy).abs()).clamp_min(0))
        elif shape_id == 3:
            thickness = max(1, image_size // 16)
            angle = rng.choice([0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0])
            dx = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
            dy = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
            mask = (dx.abs() <= thickness) | (dy.abs() <= thickness)
            mask &= (xx - cx).abs().maximum((yy - cy).abs()) <= size
        elif shape_id == 4:
            outer = (xx - cx).square() + (yy - cy).square() <= size * size
            inner = (xx - cx).square() + (yy - cy).square() <= (size * 0.55) ** 2
            mask = outer & ~inner
        else:
            dx = (xx - cx).abs()
            dy = (yy - cy).abs()
            mask = (dx + dy) <= size

        pattern_id = (label // 6) % 4
        if pattern_id == 1:
            mask &= ((xx.long() + sample_id) % max(2, image_size // 8)) < max(1, image_size // 16)
        elif pattern_id == 2:
            mask &= ((yy.long() + sample_id) % max(2, image_size // 8)) < max(1, image_size // 16)
        elif pattern_id == 3:
            mask &= (((xx.long() + yy.long() + sample_id) % max(2, image_size // 7)) < max(1, image_size // 15))

        images[i] = color.view(3, 1, 1) * mask.float().unsqueeze(0)

    images = images * 2.0 - 1.0
    return images, labels


def build_geometry_pool(args, device):
    labels = torch.arange(args.class_num, device=device).repeat_interleave(args.samples_per_class)
    sample_ids = torch.arange(args.samples_per_class, device=device).repeat(args.class_num)
    images, labels = _draw_geometry_from_labels(labels, sample_ids, args.image_size, device, args.class_num)
    return images, labels


def draw_geometry_batch(batch_size, image_size, device, class_num=4):
    labels = torch.randint(0, class_num, (batch_size,), device=device)
    sample_ids = torch.randint(0, 100000, (batch_size,), device=device)
    return _draw_geometry_from_labels(labels, sample_ids, image_size, device, class_num)


def build_model(kind, args, device):
    if args.preset == "b16":
        model_names = {
            "jit": "JiT-B/16",
            "hrm": "JiT-HRM-B/16",
            "hrm_fullgrad": "JiT-HRM-B-FullGrad/16",
            "hrm_adaln": "JiT-HRM-B-AdaLN/16",
            "hrm_adaln_fullgrad": "JiT-HRM-B-AdaLNFullGrad/16",
            "hrm_ds2": "JiT-HRM-B/16",
            "hrm_ds4": "JiT-HRM-B/16",
            "hrm_adaln_ds2": "JiT-HRM-B-AdaLN/16",
            "hrm_adaln_ds4": "JiT-HRM-B-AdaLN/16",
        }
        model_name = model_names[kind]
        deep_segments = 1
        if kind.endswith("_ds2"):
            deep_segments = 2
        elif kind.endswith("_ds4"):
            deep_segments = 4

        net = create_jit_model(
            model_name,
            input_size=args.image_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        model = TinyDenoiser(net, class_num=args.class_num, deep_segments=deep_segments)
        return model.to(device)

    if kind == "jit":
        net = JiT(
            input_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=3,
            hidden_size=128,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            num_classes=args.class_num,
            bottleneck_dim=32,
            in_context_len=0,
            in_context_start=0,
        )
    elif kind == "hrm":
        net = JiTHRM(
            input_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=3,
            hidden_size=128,
            H_layers=1,
            L_layers=1,
            H_cycles=2,
            L_cycles=2,
            num_heads=4,
            mlp_ratio=4.0,
            num_classes=args.class_num,
            bottleneck_dim=32,
            one_step_grad=True,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    model = TinyDenoiser(net, class_num=args.class_num)
    return model.to(device)


def run_one(kind, args, device, geometry_pool=None):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = build_model(kind, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    rows = []
    start = time.time()
    for step in range(1, args.steps + 1):
        if geometry_pool is None:
            images, labels = draw_geometry_batch(args.batch_size, args.image_size, device, args.class_num)
        else:
            pool_images, pool_labels = geometry_pool
            idx = torch.randint(0, pool_images.shape[0], (args.batch_size,), device=device)
            images = pool_images.index_select(0, idx)
            labels = pool_labels.index_select(0, idx)
        loss = model(images, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        rows.append({"model": kind, "step": step, "loss": loss_value})

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"{kind:>3} step {step:04d}/{args.steps}: loss={loss_value:.6f}")

    elapsed = time.time() - start
    final_window = rows[-min(20, len(rows)):]
    final_avg = sum(row["loss"] for row in final_window) / len(final_window)
    return rows, {
        "model": kind,
        "initial_loss": rows[0]["loss"],
        "final_loss": rows[-1]["loss"],
        "final_20_avg": final_avg,
        "elapsed_sec": elapsed,
        "params_m": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["tiny", "b16"], default="tiny")
    parser.add_argument("--models", default="jit,hrm",
                        help="Comma-separated models. Supports jit, hrm, hrm_fullgrad, hrm_adaln, hrm_adaln_fullgrad, hrm_ds2, hrm_ds4, hrm_adaln_ds2, hrm_adaln_ds4")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--class_num", type=int, default=4)
    parser.add_argument("--samples_per_class", type=int, default=0,
                        help="Build a fixed finite geometry pool with this many samples per class. Use 0 for online samples.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--output_dir", default="output_dir/dummy_geometry_ablation")
    args = parser.parse_args()

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"device={device}, steps={args.steps}, batch_size={args.batch_size}")

    geometry_pool = None
    if args.samples_per_class > 0:
        geometry_pool = build_geometry_pool(args, device)
        print(f"geometry_pool={geometry_pool[0].shape[0]} images ({args.class_num} classes x {args.samples_per_class})")

    all_rows = []
    summaries = []
    model_kinds = [item.strip() for item in args.models.split(",") if item.strip()]
    for kind in model_kinds:
        rows, summary = run_one(kind, args, device, geometry_pool=geometry_pool)
        all_rows.extend(rows)
        summaries.append(summary)

    losses_path = Path(args.output_dir) / "losses.csv"
    with losses_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "step", "loss"])
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = Path(args.output_dir) / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "initial_loss", "final_loss", "final_20_avg", "elapsed_sec", "params_m"])
        writer.writeheader()
        writer.writerows(summaries)

    print("\nsummary")
    for summary in summaries:
        print(
            f"{summary['model']:>3}: params={summary['params_m']:.3f}M "
            f"initial={summary['initial_loss']:.6f} final={summary['final_loss']:.6f} "
            f"final20={summary['final_20_avg']:.6f} time={summary['elapsed_sec']:.1f}s"
        )
    print(f"wrote {losses_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
