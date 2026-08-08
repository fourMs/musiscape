"""The package version has one source, and the build reads it from there.

``__version__`` drifted from the packaged version: 0.3.2 was released
reporting itself as 0.3.1, because the bump edited ``pyproject.toml`` and
left the module attribute behind. Anything citing this toolbox by version
-- a report, a deposit, a methods section -- is then citing a number the
installed package will not confirm, which makes the drift a correctness
problem rather than a cosmetic one. The same fault had run through three
ambiscape releases before it was caught there.

The fix is that ``src/musiscape/__init__.py`` holds the number and
setuptools reads it from there. These tests keep it that way.

The checks read ``pyproject.toml`` with a small section scanner rather
than a TOML parser, because ``tomllib`` is standard only from Python 3.11
and this package supports 3.10. What is needed here is whether a key is
present in a section, which does not warrant a dependency.
"""
import re
from pathlib import Path

import pytest

import musiscape

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Running against an installed wheel rather than a checkout: there is no
# pyproject.toml to inspect and nothing here applies.
pytestmark = pytest.mark.skipif(not PYPROJECT.exists(),
                                reason="no pyproject.toml (installed package)")


def _section(name: str) -> list[str]:
    """Non-comment, non-blank lines of one top-level ``[section]``."""
    out, inside = [], False
    for raw in PYPROJECT.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            inside = line == f"[{name}]"
            continue
        if inside and line and not line.startswith("#"):
            out.append(line)
    return out


def _value(section: str, key: str) -> str | None:
    for line in _section(section):
        m = re.match(rf"{re.escape(key)}\s*=\s*(.+)$", line)
        if m:
            return m.group(1).strip()
    return None


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?\w+)?", musiscape.__version__), \
        f"__version__ is not a version string: {musiscape.__version__!r}"


def test_pyproject_declares_no_second_version():
    """A static version in pyproject.toml is how the drift happened."""
    assert _value("project", "version") is None, (
        "pyproject.toml carries its own version again. It must stay dynamic, "
        "or the two numbers will drift again as they did at 0.3.2."
    )
    dynamic = _value("project", "dynamic") or ""
    assert "version" in dynamic, \
        "pyproject.toml should declare version in [project].dynamic"


def test_build_reads_the_module_attribute():
    attr = _value("tool.setuptools.dynamic", "version") or ""
    assert "musiscape.__version__" in attr, (
        f"the build resolves the version from {attr!r}; it should read "
        "musiscape.__version__ so there is exactly one place to edit"
    )


def test_setuptools_resolves_the_declared_version():
    """The number setuptools would package equals the one the module reports.

    setuptools is a build-time dependency and is absent from a plain
    runtime environment, so this cross-check skips rather than fails
    where it is not installed. The three checks above need no imports and
    carry the guard on their own.
    """
    try:
        from setuptools.config.pyprojecttoml import read_configuration
    except ImportError:
        pytest.skip("setuptools is a build-time dependency and is not "
                    "installed in this environment")

    resolved = read_configuration(str(PYPROJECT))["project"].get("version")
    assert resolved == musiscape.__version__, (
        f"build would package {resolved!r} while the module reports "
        f"{musiscape.__version__!r}"
    )
