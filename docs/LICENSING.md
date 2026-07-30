# Licensing scope

The root [MIT License](../LICENSE) applies to the original analysis code,
tests, build scripts, and repository documentation created for this
repository.

It does **not** automatically relicense measurement files copied from the
upstream measurement repository:

- upstream repository: `https://github.com/TheBuccaneer/energy`
- pinned subtree: `new`
- pinned commit: `e54128a613e7d6adc46150c020b26b6f98a4c0a2`
- local source snapshot: `data/source-snapshots/energy-e54128a613e7/`

Those measurement files retain the licensing and attribution conditions of
their source repository. Third-party Python packages retain their respective
licenses.

Before a public archival release, verify that the upstream repository has an
explicit license covering the bundled measurement snapshots. If it does not,
either add an appropriate license upstream or distribute only manifests and
reproduction instructions rather than copied data.
