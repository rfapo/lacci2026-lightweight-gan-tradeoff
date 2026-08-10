#!/usr/bin/env python3
"""Camera-ready LA-CCI (paper 69) — latencia de inferencia a b=1 e b=64.

Responde ao Revisor 1: "Batch-1 latency ... should be reported".

Mede os dois batches na MESMA execucao, com o mesmo protocolo do paper
(warmup=10, trials=50, torch.cuda.synchronize antes e depois de cada trial,
gerador em eval()). Medir b=64 junto e proposital: serve de controle, porque
so podemos publicar o b=1 se o b=64 desta sessao reproduzir os valores ja
publicados (0.68 / 3.79 / 1.21 ms/img).

StyleGAN2-Ada roda no ambiente py3.8/torch1.9 pinado; os outros dois no
ambiente py3.10/torch2.0. Sao processos distintos, mesmo host e mesma GPU
ociosa, com este mesmo codigo de temporizacao.

Uso (na .49):
  <env>/bin/python cr_latency_b1_b64.py --system organicells
  <env>/bin/python cr_latency_b1_b64.py --system fastgan64
  <sg2_env>/bin/python cr_latency_b1_b64.py --system stylegan2ada
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT_DIR = Path("/home/device002/runs/camera_ready")
WARMUP = 10
TRIALS = 50
BATCHES = [1, 64]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PUBLISHED_B64 = {"organicells": 0.68, "stylegan2ada": 3.79, "fastgan64": 1.21}


class OrganicellsGenerator(nn.Module):
    """Identica a et07_run_full.py:Generator (a que produziu os checkpoints)."""

    def __init__(self, latent_dim=100, channels=3, img_size=64, dropout=0.2):
        super().__init__()
        self.init_size = img_size // 8
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size * self.init_size),
            nn.BatchNorm1d(256 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(64, channels, 3, 1, 1), nn.Tanh(),
        )
        self.latent_dim = latent_dim

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        return self.conv_blocks(out)


def build(system: str):
    """Return (callable taking batch size, latent dim, param count in M)."""
    if system == "organicells":
        g = OrganicellsGenerator().to(DEVICE).eval()
        n = sum(p.numel() for p in g.parameters()) / 1e6
        return (lambda b: g(torch.randn(b, g.latent_dim, device=DEVICE))), n

    if system == "fastgan64":
        sys.path.insert(0, "/home/device002/FastGAN-pytorch")
        from models_64 import Generator  # noqa: E402
        g = Generator(ngf=64, nz=256, im_size=64).to(DEVICE).eval()
        n = sum(p.numel() for p in g.parameters()) / 1e6
        return (lambda b: g(torch.randn(b, 256, device=DEVICE))), n

    if system == "stylegan2ada":
        sys.path.insert(0, "/home/device002/stylegan2-ada-pytorch")
        import dnnlib  # noqa: E402
        import legacy  # noqa: E402
        snap = ("/home/device002/runs/stylegan_basophil_64/"
                "00000-dataset-mirror-auto1-gamma10-kimg300-batch16-ada/"
                "network-snapshot-000300.pkl")
        with dnnlib.util.open_url(snap) as f:
            g = legacy.load_network_pkl(f)["G_ema"].to(DEVICE)
        g.eval()
        n = sum(p.numel() for p in g.parameters()) / 1e6

        def call(b):
            z = torch.randn(b, g.z_dim, device=DEVICE)
            c = torch.zeros(b, g.c_dim, device=DEVICE) if g.c_dim > 0 else None
            return g(z, c, truncation_psi=1.0, noise_mode="const")

        return call, n

    raise SystemExit(f"sistema desconhecido: {system}")


def measure(fn, batch: int) -> dict:
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(WARMUP):
            fn(batch)
            torch.cuda.synchronize()
        per_batch = []
        for _ in range(TRIALS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(batch)
            torch.cuda.synchronize()
            per_batch.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(per_batch)
    return {
        "batch": batch,
        "ms_per_batch_mean": float(a.mean()),
        "ms_per_batch_std": float(a.std(ddof=1)),
        "ms_per_img": float(a.mean() / batch),
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True,
                    choices=["organicells", "fastgan64", "stylegan2ada"])
    ap.add_argument("--batches", default=",".join(str(b) for b in BATCHES),
                    help="lista separada por virgula; use '64' para isolar a VRAM "
                         "do efeito de ordem (medir b=1 antes muda o alocador?)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    fn, params_m = build(args.system)
    print(f"[{args.system}] params={params_m:.2f} M  device={torch.cuda.get_device_name(0)}")

    res = {"system": args.system, "params_M": params_m,
           "warmup": WARMUP, "trials": TRIALS,
           "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "measurements": {}}

    for b in batches:
        m = measure(fn, b)
        res["measurements"][f"b{b}"] = m
        print(f"  b={b:<3d} {m['ms_per_img']:8.3f} ms/img   "
              f"({m['ms_per_batch_mean']:.3f} +- {m['ms_per_batch_std']:.3f} ms/batch)   "
              f"VRAM {m['vram_peak_mb'] / 1000:.2f} GB")

    if 64 not in batches:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"latency_{args.system}{args.tag}.json").write_text(json.dumps(res, indent=2))
        return
    pub = PUBLISHED_B64[args.system]
    got = res["measurements"]["b64"]["ms_per_img"]
    res["published_b64_ms_per_img"] = pub
    res["b64_relative_deviation"] = abs(got - pub) / pub
    print(f"  controle b=64: publicado {pub:.2f}  medido {got:.2f}  "
          f"desvio {100 * res['b64_relative_deviation']:.1f}%")

    if "b1" in res["measurements"]:
        ratio = res["measurements"]["b1"]["ms_per_img"] / got
        res["b1_over_b64_ratio"] = ratio
        print(f"  razao b1/b64: {ratio:.2f}x")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"latency_{args.system}{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[ok] -> {out}")


if __name__ == "__main__":
    main()
