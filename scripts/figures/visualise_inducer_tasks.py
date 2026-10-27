"""Draw the eight inducer-pattern tasks of slow sensing (training/inducer.py): one panel per
pattern, one row per class, a few example traces of each class as a run receives them.

The traces are in units of the pool's SD (``scale: sd``), and time is in minutes (one input
step). With ``--image-delay`` the baseline after the pattern is shaded; the array is imaged at
the end of the trace.

    python scripts/figures/visualise_inducer_tasks.py
    python scripts/figures/visualise_inducer_tasks.py --condition device --image-delay 60

It writes ``results/figures/inducer/inducer-tasks-<condition>[-delay<D>].png``
(``--formats pdf,svg`` adds a PDF and an SVG).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from training import inducer
from utils.visualisation import OKABE_ITO, STYLE, add_formats_option, save_figure

TITLES = {"freq": "Pulse frequency", "ramp": "Ramp direction", "order": "Pulse order",
          "shape": "Pulse shape", "xor": "XOR of A/B pulses", "parity3": "Parity of 3 pulses",
          "xor_tr": "XOR of sawtooth pairs", "parity3_tr": "Parity of 3 sawtooth pairs"}


def examples(pattern: str, condition: str, per_class: int, image_delay: int, seed: int
             ) -> list[np.ndarray]:
    """``per_class`` traces (n, steps) of each class of ``pattern``, as a run receives them."""
    task = inducer.resolve_task({"type": "inducer", "pattern": pattern, "condition": condition,
                                 "image_delay": image_delay, "n_per_class": 20}, seed)
    X, Y, _ = inducer.load(task)
    labels = Y[:, 0].argmax(axis=1)
    return [X[labels == k][:per_class, :, 0] for k in range(Y.shape[-1])]


def draw(condition: str = "ideal", per_class: int = 3, image_delay: int = 0,
         seed: int = 6337) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True, layout="constrained")
    for ax, pattern in zip(axes.flat, inducer.PATTERNS, strict=True):
        classes = examples(pattern, condition, per_class, image_delay, seed)
        step = 1.2 * max(np.ptp(traces) for traces in classes)
        names = inducer.CLASS_NAMES[pattern]
        for k, traces in enumerate(classes):
            offset = -k * step
            for trace in traces:
                ax.plot(trace + offset, color=OKABE_ITO[k], lw=1.0, alpha=0.8)
            ax.text(1.01, offset, names[k], transform=ax.get_yaxis_transform(),
                    color=OKABE_ITO[k], fontsize=10, va="bottom")
        if image_delay:
            ax.axvspan(inducer.LENGTH, inducer.LENGTH + image_delay, color="0.9", zorder=0)
        ax.set_title(f"{TITLES[pattern]} ({pattern})", fontsize=12, loc="left")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time (min)")
    delay = f", image {image_delay} min after the pattern" if image_delay else ""
    fig.suptitle(f"Inducer-pattern tasks, {condition} delivery{delay}", fontsize=14)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw example traces of each class of the "
                                                 "eight inducer-pattern tasks.")
    parser.add_argument("--condition", default="ideal", choices=inducer.CONDITIONS,
                        help="Ideal delivery or a syringe pump (default: ideal).")
    parser.add_argument("--per-class", type=int, default=3,
                        help="Example traces of each class (default: 3).")
    parser.add_argument("--image-delay", type=int, default=0,
                        help="Minutes of baseline after the pattern (default: 0).")
    parser.add_argument("--seed", type=int, default=6337, help="Task seed (default: 6337).")
    add_formats_option(parser)
    args = parser.parse_args()

    plt.style.use(STYLE)
    fig = draw(args.condition, args.per_class, args.image_delay, args.seed)
    name = f"inducer-tasks-{args.condition}" + (f"-delay{args.image_delay}"
                                                if args.image_delay else "")
    paths = save_figure(fig, name, topic="inducer", formats=args.formats)
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
