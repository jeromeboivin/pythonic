"""
Train the model that suggests starting patches to tools/fit_samples.py.

    venv/bin/python tools/fit_prior.py <patches dir> [--patches 60000] [--epochs 15]

Random .mtdrum patches found under <patches dir> are rendered once at
velocity 127; a small CNN learns to predict, from the log-mel spectrogram of
the first second, the patch's continuous controls (in the unit-cube coding of
fit_samples.py) and its switch settings. fit_samples.py then starts from the prediction, and from
the prediction with each switch set to its second most likely value.

The renders are cached in --data (re-used unless --rebuild); the model is
written to --out, fit_prior.pt at the repository root by default, where
fit_samples.py looks for it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import pathlib
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fit_samples as F  # noqa: E402

N_FFT, HOP, N_MELS = 1024, 512, 64
SECONDS = 1.0
FRAMES = int(SECONDS * F.SR) // HOP + 1
WIN = np.hanning(N_FFT).astype(np.float32)
BANK = F.mel_bank(N_FFT, N_MELS)
N_CONT = len(F.CONTINUOUS)
N_SW = len(F.SWITCHES)


def features(x: np.ndarray) -> np.ndarray:
    """(N_MELS, FRAMES) log-mel spectrogram of the first second, 0 at -80 dB under the peak, 1 at it."""
    x = np.asarray(x[:int(SECONDS * F.SR)], np.float32)
    x = np.concatenate([x, np.zeros(int(SECONDS * F.SR) - len(x), np.float32)])
    mel = F.stft_power(x, N_FFT, HOP, WIN)[:FRAMES] @ BANK
    db = 10.0 * np.log10(mel + mel.max() * 1e-8 + 1e-30)
    return np.clip((db - db.max() + 80.0) / 80.0, 0.0, 1.0).T


def _render_chunk(files: list) -> tuple:
    parser = F.DrumPatchParser()
    feats, conts, sws = [], [], []
    for f in files:
        try:
            p = F.neutral(F.params_from_patch(parser.parse_file(f)))
            y = F.render(p, int(SECONDS * F.SR))
        except Exception:
            continue
        if not np.all(np.isfinite(y)) or np.abs(y).max() < 1e-6:
            continue
        feats.append(np.round(features(y) * 255.0).astype(np.uint8))
        conts.append(F.encode(p).astype(np.float32))
        sws.append(np.array(F.switches_of(p), np.int8))
    return feats, conts, sws


def build(library: pathlib.Path, n: int, jobs: int, seed: int, path: pathlib.Path) -> None:
    files = sorted(os.path.join(d, f) for d, _, names in os.walk(library)
                   for f in names if f.endswith('.mtdrum'))
    if not files:
        sys.exit(f'no .mtdrum patches under {library}')
    random.Random(seed).shuffle(files)
    files = files[:n]
    chunks = [files[i:i + 500] for i in range(0, len(files), 500)]
    feats, conts, sws = [], [], []
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        for k, (fe, co, sw) in enumerate(pool.map(_render_chunk, chunks)):
            feats += fe
            conts += co
            sws += sw
            if k % 10 == 9:
                print(f'  rendered {len(feats)} / {len(files)} ({time.time() - t0:.0f} s)', flush=True)
    np.savez(path, feats=np.stack(feats), conts=np.stack(conts), sws=np.stack(sws))
    print(f'{len(feats)} renders cached in {path}')


def model():
    import torch.nn as nn

    def block(a, b):
        return [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]

    class Prior(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(*block(1, 16), *block(16, 32), *block(32, 64), *block(64, 64),
                                      nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(),
                                      nn.Linear(64 * 8, 256), nn.ReLU(), nn.Dropout(0.2))
            self.cont = nn.Linear(256, N_CONT)
            self.sw = nn.Linear(256, N_SW * F.SWITCH_CHOICES)

        def forward(self, x):
            h = self.body(x[:, None])
            return self.cont(h).sigmoid(), self.sw(h).view(-1, N_SW, F.SWITCH_CHOICES)

    return Prior()


def train(data: pathlib.Path, out: pathlib.Path, epochs: int, seed: int) -> None:
    import torch
    import torch.nn.functional as tf

    torch.manual_seed(seed)
    d = np.load(data)
    x = torch.from_numpy(d['feats'])
    c = torch.from_numpy(d['conts'])
    s = torch.from_numpy(d['sws']).long()
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    n_val = max(len(x) // 20, 1)
    val, tr = perm[:n_val], perm[n_val:]
    net = model()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * math.ceil(len(tr) / 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)

    def losses(idx):
        pc, ps = net(x[idx].float() / 255.0)
        return tf.mse_loss(pc, c[idx]), tf.cross_entropy(ps.reshape(-1, F.SWITCH_CHOICES), s[idx].reshape(-1))

    for epoch in range(epochs):
        t0 = time.time()
        net.train()
        for b in tr[torch.randperm(len(tr))].split(256):
            mse, ce = losses(b)
            opt.zero_grad()
            (mse + 0.1 * ce).backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            mse = ce = acc = 0.0
            for b in val.split(1024):
                pc, ps = net(x[b].float() / 255.0)
                mse += float(tf.mse_loss(pc, c[b], reduction='sum')) / N_CONT
                acc += float((ps.argmax(-1) == s[b]).float().sum()) / N_SW
        print(f'epoch {epoch + 1:2d}: validation RMS error {math.sqrt(mse / n_val):.3f} (unit scale),'
              f' switch accuracy {acc / n_val:.1%}  ({time.time() - t0:.0f} s)', flush=True)
    torch.save({'state': net.state_dict(), 'features': [N_FFT, HOP, N_MELS, SECONDS]}, out)
    print(f'model written to {out}')


def load(path: str):
    import torch
    torch.set_num_threads(1)
    ckpt = torch.load(path, map_location='cpu')
    if ckpt['features'] != [N_FFT, HOP, N_MELS, SECONDS]:
        raise ValueError(f'{path} was trained on other features: retrain it')
    net = model()
    net.load_state_dict(ckpt['state'])
    return net.eval()


def predict(net, x: np.ndarray) -> list:
    """The predicted patch, then one per switch set to its second most likely value."""
    import torch
    with torch.no_grad():
        pc, ps = net(torch.from_numpy(features(x))[None].float())
    u, logits = pc[0].numpy(), ps[0].numpy()
    best = tuple(int(i) for i in logits.argmax(-1))
    out = [F.decode(u, best)]
    for k in range(N_SW):
        sw = list(best)
        sw[k] = int(np.argsort(-logits[k])[1])
        out.append(F.decode(u, tuple(sw)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('patches_dir', help='folder of .mtdrum patches to learn from (searched recursively)')
    ap.add_argument('--patches', type=int, default=60000, help='patches rendered for training')
    ap.add_argument('--data', default=str(F.ROOT / 'fit_prior_data.npz'), help='render cache')
    ap.add_argument('--rebuild', action='store_true', help='render again even if --data exists')
    ap.add_argument('--out', default=str(F.DEFAULT_PRIOR), help='model file')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1, help='render processes')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    if args.rebuild or not data.exists():
        build(pathlib.Path(args.patches_dir), args.patches, args.jobs, args.seed, data)
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    train(data, pathlib.Path(args.out), args.epochs, args.seed)


if __name__ == '__main__':
    main()
