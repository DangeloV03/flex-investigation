"""Write refinement job JSONs and push them onto the dispatcher queue."""

from __future__ import annotations

import json
import os
import sys

from flexsim.coex.queue_manifest import prepend_pending, read_manifest


# ---------------------------------------------------------------------------
# Job dispatching
# ---------------------------------------------------------------------------

def make_job_json(
    job_template: dict,
    mu: float,
    samples_dir: str,
    *,
    results_base: str | None = None,
    manage_csv: str | None = None,
) -> str:
    """Write a new JSON job file for a given mu and return its path."""
    scheme = str(job_template["scheme"])
    Ly = int(job_template["Ly"])
    mu_tag = f"mu{round(abs(mu) * 1_000_000):07d}"
    filename = f"{scheme}_Ly{Ly}_{mu_tag}.json"
    os.makedirs(samples_dir, exist_ok=True)
    filepath = os.path.join(samples_dir, filename)

    # Coerce all values to native Python types for JSON serialization
    def to_native(v):
        if hasattr(v, "item"):  # catches all numpy scalars
            return v.item()
        return v

    job = {k: to_native(v) for k, v in job_template.items()}
    job["mu"] = round(float(mu), 6)
    if results_base is not None:
        job["results_base"] = results_base
    if manage_csv is not None:
        job["manage_csv"] = manage_csv

    with open(filepath, "w") as f:
        json.dump(job, f, indent=2)
    return filepath


def enqueue_jobs(
    mu_values: list[float],
    job_template: dict,
    samples_dir: str,
    manifest_path: str = "run_all_queue.json",
    *,
    results_dir: str | None = None,
    manage_path: str | None = None,
) -> tuple[int, list[str]]:
    """Write JSON files and prepend them to the dispatcher queue (priority stack).

    Returns (number of paths newly added to pending, all json paths).
    """
    paths = []
    for mu in mu_values:
        json_path = make_job_json(
            job_template,
            mu,
            samples_dir,
            results_base=results_dir,
            manage_csv=manage_path,
        )
        done_json = os.path.join(samples_dir, "done", os.path.basename(json_path))
        if os.path.isfile(done_json):
            os.remove(done_json)
        paths.append(json_path)

    manifest = read_manifest(manifest_path)
    known = set(manifest.get("pending", [])) | set(manifest.get("in_flight", {}).values())

    n_added = prepend_pending(paths, path=manifest_path)

    if n_added > 0:
        for json_path, mu in zip(paths, mu_values):
            print(f"[analyzer] Enqueued {json_path} (mu={mu:.6f})")
    elif paths and not all(p in known for p in paths):
        print(
            f"[analyzer] WARNING: wrote {len(paths)} JSON(s) but added 0 to "
            f"'{manifest_path}' (paths may be duplicates in that manifest)",
            file=sys.stderr,
        )
    return n_added, paths


def jobs_already_in_manifest(paths: list[str], manifest_path: str) -> bool:
    """True when every path is already pending or in flight."""
    if not paths:
        return False
    manifest = read_manifest(manifest_path)
    known = set(manifest.get("pending", [])) | set(manifest.get("in_flight", {}).values())
    return all(p in known for p in paths)
