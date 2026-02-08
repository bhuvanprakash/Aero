# Release Process

This document describes the exact steps to cut a new AERO release.

## Prerequisites

1. All tests passing on main branch (CI green)
2. CHANGELOG.md updated with release notes
3. For PyPI publishing: `PYPI_API_TOKEN` secret configured in GitHub repository

## Steps

### 1. Prepare the Release

```bash
# Ensure you're on main and up to date
git checkout main
git pull origin main

# Run full test suite locally
cd py && python -m pytest tests/ -v
cd ../cpp && cmake -S . -B build && cmake --build build && cd ..

# Verify CLI works
python -m aerotensor.cli --version
python -m aerotensor.cli make-test-vector /tmp/test.aero
python -m aerotensor.cli validate --full /tmp/test.aero
```

### 2. Bump Version

Update version in **three places**:

**A. Python package** (`py/aerotensor/__init__.py`):
```python
__version__ = "X.Y.Z"
```

**B. C++ library** (`cpp/include/aero/aero.hpp`):
```cpp
#define AERO_VERSION_MAJOR X
#define AERO_VERSION_MINOR Y
#define AERO_VERSION_PATCH Z
```

**C. CHANGELOG.md**:
- Move `[Unreleased]` content to `[X.Y.Z] - YYYY-MM-DD`
- Add comparison link at bottom

### 3. Commit and Tag

```bash
git add py/aerotensor/__init__.py cpp/include/aero/aero.hpp CHANGELOG.md
git commit -m "Release vX.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

### 4. Push to GitHub

```bash
git push origin main
git push origin vX.Y.Z
```

This triggers the `release.yml` workflow which will:
- Build Python sdist and wheel
- Build C++ tools for Linux, macOS, Windows
- Create GitHub Release with all artifacts
- Publish to PyPI (if `PYPI_API_TOKEN` secret exists)

### 5. Verify Release

1. Check GitHub Actions: https://github.com/bhuvanprakash/Aero/actions
2. Verify GitHub Release created: https://github.com/bhuvanprakash/Aero/releases
3. If PyPI publish enabled, verify: https://pypi.org/project/aerotensor/

### 6. Announce

- Update README.md badges if needed
- Post release notes to discussions/blog/social media

## PyPI Publishing Setup (Optional)

To enable automatic PyPI publishing:

1. Create PyPI API token:
   - Go to https://pypi.org/manage/account/token/
   - Create token with scope limited to `aerotensor` project
   - Copy token (starts with `pypi-...`)

2. Add to GitHub repository secrets:
   - Go to repository Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: paste your token
   - Click "Add secret"

3. Next release will automatically publish to PyPI

## Manual PyPI Publishing (Fallback)

If automatic publishing fails or is not configured:

```bash
cd py
python -m pip install --upgrade build twine
python -m build
python -m twine upload dist/aerotensor-X.Y.Z*
```

## Hotfix Releases

For urgent fixes on a released version:

```bash
# Create hotfix branch from tag
git checkout -b hotfix/vX.Y.Z+1 vX.Y.Z

# Make fixes, update CHANGELOG
# ... commit fixes ...

# Bump patch version and tag
git tag -a vX.Y.Z+1 -m "Hotfix vX.Y.Z+1"
git push origin vX.Y.Z+1

# Merge back to main
git checkout main
git merge hotfix/vX.Y.Z+1
git push origin main
```

## Troubleshooting

### Release workflow fails

1. Check GitHub Actions logs for specific error
2. Common issues:
   - Missing dependencies in workflow
   - Test failures (fix and re-tag)
   - PyPI token expired (regenerate and update secret)

### PyPI publish fails

1. Check if version already exists on PyPI (cannot overwrite)
2. Verify token has correct permissions
3. Check `twine upload` output for specific error

### C++ tools build fails

1. Check matrix OS logs in release workflow
2. Common issues:
   - CMake version mismatch
   - Missing system dependencies
   - Compiler compatibility

## Version Numbering

AERO follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible format changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

Examples:
- `0.1.0` → `0.1.1`: Bug fixes, hardening
- `0.1.1` → `0.2.0`: Remote range fetching (new feature)
- `0.2.0` → `1.0.0`: Format breaking change (avoid if possible)
