# Command-line interface.
#
#   python -m stochaxtic run   out.npz [options]
#   python -m stochaxtic merge out.npz part1.npz part2.npz ...
#
# Hyperparameters are passed as JSON, e.g.
#   --mass-model bpl_double_gaussian --mass-kwargs '{"params": {"alpha_m_1": 2.0}}'
#   --rate-model madau_dickinson_timedelay --rate-kwargs '{"R0": 23.4}'

import argparse
import json
import os
import sys

import numpy as np

from .integrator import OmegaIntegrator, OmegaResult


def _parser():
    p = argparse.ArgumentParser(prog="python -m stochaxtic",
                                description="Omega_GW(f) of a compact-binary population")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="compute Omega_GW and save it to an .npz")
    r.add_argument("output", help="output .npz file (skipped if it exists)")
    r.add_argument("--mass-model", default="bpl_double_gaussian")
    r.add_argument("--mass-kwargs", type=json.loads, default={})
    r.add_argument("--rate-model", default="madau_dickinson")
    r.add_argument("--rate-kwargs", type=json.loads, default={})
    r.add_argument("--R0", type=float, default=None,
                   help="local rate in Gpc^-3 yr^-1 (shortcut for rate-kwargs R0)")
    r.add_argument("--waveform", default="IMRPhenomXAS")
    r.add_argument("--waveform-kwargs", type=json.loads, default={},
                   help='e.g. \'{"mode": "fd"}\' or \'{"isco_cut": 3}\'')
    r.add_argument("--fmin", type=float, default=5.0)
    r.add_argument("--fmax", type=float, default=1000.0)
    r.add_argument("--df", type=float, default=1.0)
    r.add_argument("--z-min", type=float, default=0.2)
    r.add_argument("--z-max", type=float, default=10.0)
    r.add_argument("--n-z", type=int, default=1000)
    r.add_argument("--proposal-kwargs", type=json.loads, default={},
                   help='e.g. \'{"eps": 0.1, "tilts": [-3, 0, 4], "iota_power": 2}\'')
    r.add_argument("--n-rep", type=int, default=8)
    r.add_argument("--rep-start", type=int, default=0)
    r.add_argument("--log2-points", type=int, default=12)
    r.add_argument("--seed", type=int, default=20260929)
    r.add_argument("--antithetic", action="store_true")
    r.add_argument("--n-proc", type=int, default=1)

    m = sub.add_parser("merge", help="merge results with disjoint replica ranges")
    m.add_argument("output")
    m.add_argument("inputs", nargs="+")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)

    if args.cmd == "merge":
        res = OmegaResult.merge([OmegaResult.load(f) for f in sorted(args.inputs)])
        res.save(args.output)
        print(res.summary())
        print(f"[done] {args.output}")
        return

    if os.path.exists(args.output):
        print(f"[skip] {args.output} exists")
        sys.exit(0)
    if args.R0 is not None:
        args.rate_kwargs["R0"] = args.R0
    freqs = np.arange(args.fmin, args.fmax + 0.5 * args.df, args.df)
    oi = OmegaIntegrator(args.mass_model, args.rate_model, args.waveform, freqs=freqs,
                         mass_kwargs=args.mass_kwargs, rate_kwargs=args.rate_kwargs,
                         waveform_kwargs=args.waveform_kwargs,
                         z_min=args.z_min, z_max=args.z_max, n_z=args.n_z,
                         proposal_kwargs=args.proposal_kwargs, n_proc=args.n_proc)
    res = oi.run(n_rep=args.n_rep, log2_points=args.log2_points, seed=args.seed,
                 rep_start=args.rep_start, antithetic=args.antithetic)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    res.save(args.output)
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
