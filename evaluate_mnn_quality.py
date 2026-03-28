#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
import MNN

ROOT = Path(__file__).resolve().parent

@dataclass
class Metrics:
    max_abs_err: float
    mean_abs_err: float
    rmse: float
    mean_rel_err: float
    pass_fp16: bool

@dataclass
class PerfResult:
    avg_ms: float
    p95_ms: float
    max_ms: float
    stable: bool

def _load_cfg() -> dict:
    with (ROOT / "configs" / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_state(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")

def _run_mnn_single(model_path: Path, input_name: str, x: np.ndarray) -> Dict[str, np.ndarray]:
    net = MNN.Interpreter(str(model_path))
    session = net.createSession()
    inp = net.getSessionInput(session, input_name)
    net.resizeTensor(inp, tuple(x.shape))
    net.resizeSession(session)
    host_in = MNN.Tensor(tuple(x.shape), MNN.Halide_Type_Float, x.astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    inp.copyFrom(host_in)
    net.runSession(session)
    outputs = net.getSessionOutputAll(session)
    out = {}
    for name, t in outputs.items():
        shape = tuple(int(max(1, s)) for s in t.getShape())
        host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.zeros(shape, dtype=np.float32), MNN.Tensor_DimensionType_Caffe)
        t.copyToHostTensor(host)
        out[name] = np.array(host.getData(), dtype=np.float32).reshape(shape)
    return out

def _calc_metrics(y_mnn: np.ndarray, y_pt: np.ndarray) -> Metrics:
    y_mnn = y_mnn.astype(np.float64)
    y_pt = y_pt.astype(np.float64)
    d = y_mnn - y_pt
    max_abs = float(np.max(np.abs(d)))
    mean_abs = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    rel = float(np.mean(np.abs(d) / (np.abs(y_pt) + 1e-8)))
    passed = (max_abs < 1e-2) and (mean_abs < 1e-4) and (rmse < 1e-4)
    return Metrics(max_abs, mean_abs, rmse, rel, passed)

def _stability_perf(model_path: Path, input_name: str, shape: Tuple[int, ...], runs: int, seed: int) -> PerfResult:
    rng = np.random.default_rng(seed)
    net = MNN.Interpreter(str(model_path))
    session = net.createSession()
    inp = net.getSessionInput(session, input_name)
    net.resizeTensor(inp, tuple(shape))
    net.resizeSession(session)
    lat = []
    stable = True
    for _ in range(runs):
        x = rng.standard_normal(np.prod(shape), dtype=np.float32).reshape(shape)
        inp.copyFrom(MNN.Tensor(tuple(shape), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe))
        t0 = time.perf_counter()
        net.runSession(session)
        lat.append((time.perf_counter() - t0) * 1000.0)
        outs = net.getSessionOutputAll(session)
        for _, t in outs.items():
            tshape = tuple(int(max(1, s)) for s in t.getShape())
            host = MNN.Tensor(tshape, MNN.Halide_Type_Float, np.zeros(tshape, dtype=np.float32), MNN.Tensor_DimensionType_Caffe)
            t.copyToHostTensor(host)
            arr = np.array(host.getData(), dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                stable = False
                break
        if not stable:
            break
    arr = np.array(lat, dtype=np.float64)
    return PerfResult(float(arr.mean()), float(np.percentile(arr, 95)), float(arr.max()), stable)

def eval_all(cfg: dict, mnn_dir: Path, seed: int, runs: int) -> dict:
    from models.stgcn import STGCN
    from models.fno import FNO1d
    from models.risk_model import RiskMLP

    # STGCN
    stgcn = STGCN(num_nodes=cfg['stgcn']['num_nodes'], in_channels=cfg['stgcn']['in_channels'], hidden_channels=cfg['stgcn']['hidden_channels'], num_frames=cfg['stgcn']['num_frames'], output_dim=cfg['stgcn']['output_dim'], dropout=cfg['stgcn']['dropout']).eval()
    stgcn.load_state_dict(_load_state(ROOT / 'checkpoints' / 'stgcn_bestgrf_PhaseA.pth'), strict=True)
    x_stg = np.random.default_rng(seed).standard_normal((1, cfg['stgcn']['num_frames'], cfg['stgcn']['num_nodes'], 3), dtype=np.float32)
    with torch.no_grad():
        y_stg_pt, _ = stgcn(torch.from_numpy(x_stg))
    y_stg_pt = y_stg_pt.cpu().numpy()
    y_stg_mnn = _run_mnn_single(mnn_dir / 'stgcn_phaseA.mnn', 'visual_seq', x_stg)['joint_angles'].reshape(y_stg_pt.shape)

    # FNO
    fno = FNO1d(input_dim=cfg['fno']['input_dim'], output_dim=cfg['fno']['output_dim'], future_k=cfg['fno']['future_k'], modes=cfg['fno']['modes'], width=cfg['fno']['width'], depth=cfg['fno']['depth'], seq_len=cfg['fno']['seq_len'], use_lstm=True).eval()
    fno.load_state_dict(_load_state(ROOT / 'checkpoints' / 'fno_bestgrf_PhaseA.pth'), strict=False)
    x_fno = np.random.default_rng(seed + 1).standard_normal((1, cfg['fno']['seq_len'], cfg['fno']['input_dim']), dtype=np.float32)
    with torch.no_grad():
        y_fno_pt, _ = fno(torch.from_numpy(x_fno))
    y_fno_pt = y_fno_pt.cpu().numpy()
    y_fno_mnn = _run_mnn_single(mnn_dir / 'fno_lstm_phaseA.mnn', 'bio_seq', x_fno)['grf_seq'].reshape(y_fno_pt.shape)

    # Risk
    risk = RiskMLP(input_dim=cfg['risk']['input_dim'], hidden_dim=cfg['risk']['hidden_dim'], num_classes=cfg['risk']['num_classes'], seq_len=cfg['risk']['seq_len'], use_physio=cfg['risk'].get('use_physio', True)).eval()
    risk.load_state_dict(_load_state(ROOT / 'checkpoints' / 'risk_bestgrf_PhaseA.pth'), strict=True)
    x_risk = np.random.default_rng(seed + 2).standard_normal((1, cfg['risk']['seq_len'], cfg['risk']['input_dim']), dtype=np.float32)
    with torch.no_grad():
        hr = torch.tensor([[70.0]], dtype=torch.float32)
        sl = torch.tensor([[80.0]], dtype=torch.float32)
        logits_pt, conf_pt = risk(torch.from_numpy(x_risk), hr, sl)
        score_pt = torch.softmax(logits_pt, dim=-1)[:, 2:3]
    out_risk = _run_mnn_single(mnn_dir / 'risk_phaseA.mnn', 'risk_seq', x_risk)
    logits_mnn = out_risk['risk_logits'].reshape(logits_pt.shape)
    conf_mnn = out_risk['risk_confidence'].reshape(conf_pt.shape)
    score_mnn = np.exp(logits_mnn) / np.sum(np.exp(logits_mnn), axis=-1, keepdims=True)
    score_mnn = score_mnn[:, 2:3]

    res = {
        'numeric': {
            'stgcn': {'shape_match': list(y_stg_mnn.shape) == list(y_stg_pt.shape), 'metrics': asdict(_calc_metrics(y_stg_mnn, y_stg_pt)), 'functional_ok': bool(np.isfinite(y_stg_mnn).all())},
            'fno': {'shape_match': list(y_fno_mnn.shape) == list(y_fno_pt.shape), 'metrics': asdict(_calc_metrics(y_fno_mnn, y_fno_pt)), 'functional_ok': bool(np.isfinite(y_fno_mnn).all())},
            'risk': {
                'shape_match': {'logits': list(logits_mnn.shape) == list(logits_pt.shape), 'confidence': list(conf_mnn.shape) == list(conf_pt.shape), 'score': list(score_mnn.shape) == list(score_pt.shape)},
                'metrics': {'logits': asdict(_calc_metrics(logits_mnn, logits_pt.cpu().numpy())), 'confidence': asdict(_calc_metrics(conf_mnn, conf_pt.cpu().numpy())), 'risk_score': asdict(_calc_metrics(score_mnn, score_pt.cpu().numpy()))},
                'functional_ok': bool(np.isfinite(logits_mnn).all() and np.isfinite(conf_mnn).all() and ((score_mnn >= 0).all() and (score_mnn <= 1).all()))
            }
        },
        'stability_perf': {
            'stgcn': asdict(_stability_perf(mnn_dir / 'stgcn_phaseA.mnn', 'visual_seq', (1, cfg['stgcn']['num_frames'], cfg['stgcn']['num_nodes'], 3), runs, seed)),
            'fno': asdict(_stability_perf(mnn_dir / 'fno_lstm_phaseA.mnn', 'bio_seq', (1, cfg['fno']['seq_len'], cfg['fno']['input_dim']), runs, seed + 1)),
            'risk': asdict(_stability_perf(mnn_dir / 'risk_phaseA.mnn', 'risk_seq', (1, cfg['risk']['seq_len'], cfg['risk']['input_dim']), runs, seed + 2))
        }
    }
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mnn_dir', default=str(ROOT / 'export' / 'mnn_phasea_android_opt'))
    ap.add_argument('--runs', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--report', default=str(ROOT / 'logs' / 'mnn_quality_report.json'))
    args = ap.parse_args()

    cfg = _load_cfg()
    res = eval_all(cfg, Path(args.mnn_dir), args.seed, args.runs)
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with rp.open('w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'Saved report: {rp}')
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
