# Git-Centered Development and Jetson Deployment

This repo uses Git as the source of truth for source code. The Jetson should
normally run stable, validated snapshots pulled from Git. Use `scp` or `rsync`
for datasets, logs, and occasional emergency file copies, but do not treat the
Jetson as a separate source-of-truth codebase.

## Branch Model

- `main`: stable integration branch
- `feature/<short-purpose>`: active feature work
- `fix/<short-purpose>`: targeted bug fixes
- `jetson-YYYYMMDD-<short-purpose>`: annotated tags for Jetson-tested snapshots

## Normal Development Flow

1. Start new work from `main`.
2. Create a feature branch.
3. Commit locally in small, intentional checkpoints.
4. Push the branch to GitHub.
5. Test locally and, when needed, on the Jetson.
6. Merge validated work into `main`.
7. Tag Jetson-safe releases after hardware validation.

Example:

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/my-next-change
```

## Saving Progress While Continuing Development

When the project reaches a useful checkpoint but is not ready for `main` yet:

```bash
git status
git add <files>
git commit -m "Describe the checkpoint"
git push -u origin HEAD
```

Continue working on the same feature branch until the work is ready to merge.

## Jetson Deployment

Keep a real Git clone on the Jetson at `~/jetson-nano-racer`.

Preferred deployment path:

1. Commit locally on the laptop.
2. Push the branch or tag to GitHub.
3. On the Jetson, fetch the latest refs.
4. Deploy a stable target.

### Deploy a Jetson-Tested Tag

```bash
cd ~/jetson-nano-racer
git fetch --all --tags
git checkout tags/<tag> -b jetson-<tag>
```

### Deploy the Current Stable `main`

```bash
cd ~/jetson-nano-racer
git switch main
git pull --ff-only origin main
```

### Temporarily Test a Feature Branch on the Jetson

Only do this deliberately, and return the Jetson to a stable tag or `main`
after the test:

```bash
cd ~/jetson-nano-racer
git fetch --all --tags
git switch feature/<branch-name>
git pull --ff-only origin feature/<branch-name>
```

After the test:

```bash
git switch main
git pull --ff-only origin main
```

## Data Transfer

Use `scp` or `rsync` for recorded runs and artifacts.

Examples:

```bash
scp -r jetson@100.120.5.55:~/jetson-nano-racer/jetracer/train/runs_rgb_depth/run_20260325_195709 ~/Desktop/
rsync -av jetson@100.120.5.55:~/jetson-nano-racer/jetracer/train/runs_rgb_depth/ ~/Desktop/runs_rgb_depth/
```

Do not commit recorded runs, logs, caches, or machine-local artifacts.

## Jetson Drift Rule

If you hotfix a source file on the Jetson with `scp`, immediately do one of
these:

- commit the same change into Git and redeploy from Git
- discard the hotfix and restore the Jetson to a known Git ref

Do not leave the Jetson on a one-off code state that only exists on the device.
