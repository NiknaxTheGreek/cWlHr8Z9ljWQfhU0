from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(sys.argv[1]).resolve()
CLEAN_RUN_ID = 35273484952
CLEAN_RUN_HEAD = "6dd522e0b9a002a2f6267ad10480e14fd1dc8319"
CLEAN_RUN_ARTIFACT_ID = 10518614089
CLEAN_RUN_ARTIFACT_SHA256 = "3dddc022a8717cbca3b6ed32b262e40c61a78127393409720b3eb1b94801ffc2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_dir(name: str) -> Path:
    direct = ARTIFACT_ROOT / name
    if direct.is_dir():
        return direct
    matches = [p for p in ARTIFACT_ROOT.rglob(name) if p.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name!r} directory in artifact, found {matches}")
    return matches[0]


outputs_src = locate_dir("outputs")
artifacts_src = locate_dir("artifacts")
executed_notebook = artifacts_src / "01_Potential_Talents_Main.executed.ipynb"
if not executed_notebook.is_file():
    raise FileNotFoundError(executed_notebook)

# Verify the clean-run output package before publishing it.
manifest_path = outputs_src / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for relative_path, expected_hash in manifest["output_sha256"].items():
    artifact_file = ARTIFACT_ROOT / relative_path
    if not artifact_file.is_file():
        # Accommodate artifact download layouts while still validating the exact named file.
        artifact_file = outputs_src / Path(relative_path).relative_to("outputs")
    actual_hash = sha256_file(artifact_file)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Hash mismatch for {relative_path}: {actual_hash} != {expected_hash}"
        )

# Publish the exact executed notebook and exact generated outputs.
shutil.copy2(executed_notebook, ROOT / "01_Potential_Talents_Main.ipynb")
outputs_dst = ROOT / "outputs"
if outputs_dst.exists():
    shutil.rmtree(outputs_dst)
shutil.copytree(outputs_src, outputs_dst)

# Align the approved README with the filenames produced by the authoritative run.
readme_path = ROOT / "README.md"
readme = readme_path.read_text(encoding="utf-8")
replacements = {
    "outputs/figures/F03_reference_vs_predicted_rank.png": "outputs/figures/F03_target_vs_oof_rank.png",
    "outputs/figures/F04_feedback_ndcg_trajectory.png": "outputs/figures/F04_feedback_generalization.png",
    "outputs/figures/F05_management_effort_vs_gain.png": "outputs/figures/F05_management_effort_vs_fitted_gain.png",
}
for old, new in replacements.items():
    if old not in readme:
        raise RuntimeError(f"Expected README path not found: {old}")
    readme = readme.replace(old, new)
readme_path.write_text(readme, encoding="utf-8")

# Reconcile release registry to the actual authoritative outputs.
registry_path = ROOT / "project_control" / "project_registry.json"
registry = json.loads(registry_path.read_text(encoding="utf-8"))
figure_files = {
    "F01": "outputs/figures/F01_population_flow.png",
    "F02": "outputs/figures/F02_oof_prediction_vs_target.png",
    "F03": "outputs/figures/F03_target_vs_oof_rank.png",
    "F04": "outputs/figures/F04_feedback_generalization.png",
    "F05": "outputs/figures/F05_management_effort_vs_fitted_gain.png",
}
for figure in registry["figure_registry"]:
    figure["file"] = figure_files[figure["id"]]

baseline_freeze = registry["schema_registry"]["baseline_freeze"]
baseline_freeze.pop("artifact", None)
baseline_freeze["persistence"] = (
    "in-notebook analytical object; not a standalone release artifact"
)

release_paths = [
    "README.md",
    "00_Project_Audit_and_Setup.ipynb",
    "01_Potential_Talents_Main.ipynb",
    "outputs/final_ranking.csv",
    "outputs/feedback_summary.csv",
    "outputs/run_manifest.json",
    "outputs/run.log",
    "outputs/figures/F01_population_flow.png",
    "outputs/figures/F02_oof_prediction_vs_target.png",
    "outputs/figures/F03_target_vs_oof_rank.png",
    "outputs/figures/F04_feedback_generalization.png",
    "outputs/figures/F05_management_effort_vs_fitted_gain.png",
    "requirements.txt",
    "requirements-lock.txt",
]
registry["release_registry"] = [
    {"release_id": f"R{index:02d}", "path": path, "required": True}
    for index, path in enumerate(release_paths, start=1)
]
registry_path.write_text(
    json.dumps(registry, ensure_ascii=False, separators=(",", ":")) + "\n",
    encoding="utf-8",
)

# Record the final approved clean-run evidence and request the true fresh-clone gate.
state_path = ROOT / "project_control" / "execution_state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
state["current_checkpoint"] = "CP7_FINAL_RELEASE_ARTIFACTS_PUBLISHED"
state["validation_evidence"]["final_approved_clean_run"] = {
    "status": "PASS",
    "workflow_run_id": CLEAN_RUN_ID,
    "head_sha": CLEAN_RUN_HEAD,
    "tests": "PASS (same 28-test suite)",
    "model_fetch_validation": "PASS",
    "notebook_execution": "PASS",
    "artifact_id": CLEAN_RUN_ARTIFACT_ID,
    "artifact_sha256": CLEAN_RUN_ARTIFACT_SHA256,
}
release_validation = state["release_validation"]
release_validation["final_approved_clean_run_status"] = "PASS"
release_validation["final_approved_clean_run_id"] = CLEAN_RUN_ID
release_validation["final_approved_clean_run_head_sha"] = CLEAN_RUN_HEAD
release_validation["fresh_clone_status"] = "REQUESTED"
release_validation["reviewer_ready"] = False
state["handoffs"] = [
    "Validate the published final release artifacts from an explicit fresh clone.",
    "Set REVIEWER_READY = TRUE only after the fresh-clone validation passes.",
]
state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

# Final structural release checks.
for release in registry["release_registry"]:
    path = ROOT / release["path"]
    if release["required"] and not path.is_file():
        raise FileNotFoundError(f"Required release artifact missing: {release['path']}")

notebook = json.loads((ROOT / "01_Potential_Talents_Main.ipynb").read_text(encoding="utf-8"))
if len(notebook["cells"]) != 58:
    raise RuntimeError(f"Expected 58 cells in final notebook, got {len(notebook['cells'])}")
if not any(cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code"):
    raise RuntimeError("Final main notebook is not the executed notebook.")

# Publication scaffolding must not survive the release commit.
for temp in [
    ROOT / ".github" / "workflows" / "publish-final-release.yml",
    ROOT / "scripts" / "publish_final_release.py",
    ROOT / "project_control" / "publish_final_release.trigger",
]:
    if temp.exists():
        temp.unlink()

print("Final approved clean-run artifacts published and verified.")
print("Fresh-clone validation is the only remaining release gate.")
