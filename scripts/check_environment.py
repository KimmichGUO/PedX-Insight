"""Report dependency drift without importing detectors, downloading weights, or changing packages."""
import argparse
from importlib import import_module, metadata
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

CV2_PROVIDERS = ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless")
IMPORTS = ("numpy", "cv2", "pandas", "scipy", "sklearn", "statsmodels.api", "torch",
           "torchvision", "supervision", "transformers", "onnxruntime")


def inspect_environment(requirements_path=None, check_imports=False):
    requirements_path = Path(requirements_path) if requirements_path else Path(__file__).resolve().parents[1] / "requirements.txt"
    errors, versions = [], {}
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            value = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            value = None
        versions[requirement.name] = value
        try:
            if not value:
                errors.append(f"{requirement.name}: missing distribution or invalid version metadata")
            elif Version(value) not in requirement.specifier:
                errors.append(f"{requirement.name}: installed {value}, requires {requirement.specifier}")
        except InvalidVersion:
            errors.append(f"{requirement.name}: invalid installed version {value!r}")

    providers = []
    for name in CV2_PROVIDERS:
        try:
            metadata.version(name)
            providers.append(name)
        except metadata.PackageNotFoundError:
            pass
    if providers != ["opencv-python"]:
        errors.append(f"cv2 must have exactly one provider (opencv-python), found {providers}")

    imported = []
    if check_imports:
        for name in IMPORTS:
            try:
                module = import_module(name)
                imported.append(name)
                if name == "cv2":
                    for feature in ("FaceDetectorYN", "calcOpticalFlowPyrLK", "dnn", "VideoWriter"):
                        if not hasattr(module, feature):
                            errors.append(f"cv2 missing core feature {feature}")
            except Exception as e:
                errors.append(f"{name}: import failed ({type(e).__name__}: {e})")
    return {"python": sys.version.split()[0], "distribution_versions": versions,
            "cv2_providers": providers, "imported": imported, "errors": errors,
            "scope": "Dependency/import checks only; detector weights and validity are not tested"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true", help="Import scientific/CV runtimes without loading model assets")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any check fails (CI)")
    args = parser.parse_args()
    report = inspect_environment(check_imports=args.imports)
    print(json.dumps(report, indent=2))
    if args.strict and report["errors"]:
        sys.exit(1)
