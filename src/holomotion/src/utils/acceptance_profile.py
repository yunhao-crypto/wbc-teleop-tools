"""Named software acceptance targets. Never infer a pass from absent evidence."""

from pathlib import Path

from omegaconf import OmegaConf


def load_acceptance_profile(path: str | Path) -> dict:
    profile = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("Unsupported acceptance profile")
    if not isinstance(profile.get("name"), str) or not profile["name"]:
        raise ValueError("Acceptance profile requires a nonempty name")
    for section in ("asset", "collision", "physics", "observation"):
        if not isinstance(profile.get(section), dict):
            raise ValueError(f"Acceptance profile missing section: {section}")
    return profile
