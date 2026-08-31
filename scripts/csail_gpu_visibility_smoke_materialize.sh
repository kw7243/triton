#!/usr/bin/env bash
set -euo pipefail

# Materialize and prove an ordinary, independent repository before Slurm use.

umask 077

attempt=${SMOKE_ATTEMPT_ROOT:?SMOKE_ATTEMPT_ROOT is required}
bundle=${SMOKE_SOURCE_BUNDLE:?SMOKE_SOURCE_BUNDLE is required}
stage=${SMOKE_STAGE_PATH:?SMOKE_STAGE_PATH is required}
branch=${SMOKE_SOURCE_BRANCH:?SMOKE_SOURCE_BRANCH is required}
head=${SMOKE_SOURCE_HEAD:?SMOKE_SOURCE_HEAD is required}
base=${SMOKE_LINEAGE_BASE:?SMOKE_LINEAGE_BASE is required}
source_worktree=${SMOKE_SOURCE_WORKTREE:?SMOKE_SOURCE_WORKTREE is required}
source_git_dir=${SMOKE_SOURCE_GIT_DIR:?SMOKE_SOURCE_GIT_DIR is required}
source_common_dir=${SMOKE_SOURCE_COMMON_DIR:?SMOKE_SOURCE_COMMON_DIR is required}

preflight=$attempt/preflight
stage_parent=$attempt/stage
verify_repo=$preflight/bundle-verifier-repo
source_manifest=$preflight/source-tracked.manifest.nul
source_status=$preflight/source-status.porcelain-v1.nul
source_untracked=$preflight/source-ordinary-untracked.manifest.nul
source_allowed_ignored=$preflight/source-allowed-ignored.manifest.nul
source_tree_record=$preflight/source-tree.txt
stage_manifest=$preflight/stage-tracked.manifest.nul
stage_status_before=$preflight/stage-status-before-metadata.porcelain-v1.nul
stage_untracked_before=$preflight/stage-ordinary-untracked-before-metadata.manifest.nul
stage_ignored_before=$preflight/stage-ignored-before-metadata.manifest.nul
stage_status_after=$preflight/stage-status-after-metadata.porcelain-v1.nul

[ "$(id -un)" = kwen1 ]
[ -d "$attempt" ] && [ ! -L "$attempt" ]
[ -d "$preflight" ] && [ ! -L "$preflight" ]
[ -d "$attempt/results" ] && [ ! -L "$attempt/results" ]
[ -d "$attempt/run-state" ] && [ ! -L "$attempt/run-state" ]
[ -f "$bundle" ] && [ ! -L "$bundle" ]
[ -f "$source_manifest" ] && [ ! -L "$source_manifest" ]
[ -f "$source_status" ] && [ ! -L "$source_status" ]
[ -f "$source_untracked" ] && [ ! -L "$source_untracked" ]
[ -f "$source_allowed_ignored" ] && [ ! -L "$source_allowed_ignored" ]
[ -f "$source_tree_record" ] && [ ! -L "$source_tree_record" ]
[ ! -s "$source_status" ]
[ ! -s "$source_untracked" ]
[ ! -s "$source_allowed_ignored" ]
[ ! -e "$verify_repo" ] && [ ! -L "$verify_repo" ]
[ ! -e "$stage_parent" ] && [ ! -L "$stage_parent" ]
[ ! -e "$stage" ] && [ ! -L "$stage" ]

git init -q "$verify_repo"
[ -d "$verify_repo/.git" ] && [ ! -L "$verify_repo/.git" ]
(
  cd "$verify_repo"
  printf 'cwd=%s\n' "$PWD"
  printf 'is_inside=%s\n' "$(git rev-parse --is-inside-work-tree)"
  printf 'command=git bundle verify %s\n' "$bundle"
  git bundle verify "$bundle"
) > "$preflight/bundle-verify.txt" 2>&1

git bundle list-heads "$bundle" > "$preflight/bundle-heads.txt"
[ "$(awk -v ref="refs/heads/$branch" '$2 == ref {print $1}' "$preflight/bundle-heads.txt")" = "$head" ]

mkdir -m 0700 "$stage_parent"
git clone --quiet --branch "$branch" "$bundle" "$stage"

[ -d "$stage/.git" ] && [ ! -L "$stage/.git" ]
[ "$(git -C "$stage" rev-parse --path-format=absolute --git-dir)" = "$stage/.git" ]
[ "$(git -C "$stage" rev-parse --path-format=absolute --git-common-dir)" = "$stage/.git" ]
[ "$(git -C "$stage" rev-parse --verify HEAD)" = "$head" ]
[ "$(git -C "$stage" rev-parse --verify "refs/heads/$branch")" = "$head" ]
[ "$(git -C "$stage" rev-parse --verify 'HEAD^{tree}')" = "$(tr -d '\n' < "$source_tree_record")" ]
git -C "$stage" merge-base --is-ancestor "$base" "$head"
git -C "$stage" fsck --full --no-dangling > "$preflight/stage-fsck.txt" 2>&1
[ ! -s "$stage/.git/objects/info/alternates" ]
[ -z "$(find "$stage" ! -user kwen1 -print -quit)" ]

git -C "$stage" ls-files --stage -z > "$stage_manifest"
git -C "$stage" status --porcelain=v1 -z --untracked-files=all > "$stage_status_before"
git -C "$stage" ls-files --others --exclude-standard -z > "$stage_untracked_before"
git -C "$stage" ls-files --others --ignored --exclude-standard -z > "$stage_ignored_before"
cmp -- "$source_manifest" "$stage_manifest"
cmp -- "$source_status" "$stage_status_before"
cmp -- "$source_untracked" "$stage_untracked_before"
cmp -- "$source_allowed_ignored" "$stage_ignored_before"

bundle_hold=$preflight/.source.bundle.independence-hold
[ ! -e "$bundle_hold" ] && [ ! -L "$bundle_hold" ]
mv -- "$bundle" "$bundle_hold"
{
  printf 'bundle_temporarily_absent=true\n'
  git -C "$stage" fsck --full --no-dangling
  git -C "$stage" cat-file -e "$head^{commit}"
  git -C "$stage" cat-file -e "$head^{tree}"
  printf 'stage_head=%s\n' "$(git -C "$stage" rev-parse HEAD)"
  printf 'stage_git_dir=%s\n' "$(git -C "$stage" rev-parse --path-format=absolute --git-dir)"
  printf 'stage_common_dir=%s\n' "$(git -C "$stage" rev-parse --path-format=absolute --git-common-dir)"
  if [ -f "$stage/.git/objects/info/alternates" ]; then
    printf 'alternates_bytes=%s\n' "$(wc -c < "$stage/.git/objects/info/alternates")"
  else
    printf 'alternates_bytes=0\n'
  fi
} > "$preflight/independence-check.txt" 2>&1
mv -- "$bundle_hold" "$bundle"

source_manifest_sha=$(sha256sum "$source_manifest" | awk '{print $1}')
stage_manifest_sha=$(sha256sum "$stage_manifest" | awk '{print $1}')
source_status_sha=$(sha256sum "$source_status" | awk '{print $1}')
source_untracked_sha=$(sha256sum "$source_untracked" | awk '{print $1}')
source_allowed_ignored_sha=$(sha256sum "$source_allowed_ignored" | awk '{print $1}')
source_tree=$(tr -d '\n' < "$source_tree_record")
bundle_sha=$(sha256sum "$bundle" | awk '{print $1}')
payload_sha=$(sha256sum "$stage/scripts/csail_gpu_visibility_smoke.sh" | awk '{print $1}')
helper_sha=$(sha256sum "$stage/scripts/csail_gpu_visibility_smoke_materialize.sh" | awk '{print $1}')
tracked_entries=$(git -C "$stage" ls-files | wc -l)

cat > "$stage/REPRODUCIBILITY_METADATA.json" <<EOF
{
  "schema_version": "csail-gpu-visibility-stage-v2",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source_worktree": "$source_worktree",
  "source_git_dir": "$source_git_dir",
  "source_git_common_dir": "$source_common_dir",
  "source_branch": "$branch",
  "source_head": "$head",
  "source_tree": "$source_tree",
  "lineage_base": "$base",
  "staged_repo": "$stage",
  "staged_git_dir": "$stage/.git",
  "staged_git_common_dir": "$stage/.git",
  "tracked_entries": $tracked_entries,
  "source_tracked_manifest_sha256": "$source_manifest_sha",
  "stage_tracked_manifest_sha256": "$stage_manifest_sha",
  "source_status_sha256": "$source_status_sha",
  "source_ordinary_untracked_manifest_sha256": "$source_untracked_sha",
  "source_allowed_ignored_manifest_sha256": "$source_allowed_ignored_sha",
  "source_bundle_sha256": "$bundle_sha",
  "payload_sha256": "$payload_sha",
  "materializer_sha256": "$helper_sha",
  "dirty_tracked_entries": 0,
  "ordinary_untracked_entries": 0,
  "allowed_ignored_input_entries": 0,
  "excluded_required_inputs": [],
  "exclusions": ["source linked-worktree administrative .git pointer/common metadata replaced by an ordinary independent bundle clone"]
}
EOF
chmod 0444 "$stage/REPRODUCIBILITY_METADATA.json"

git -C "$stage" status --porcelain=v1 -z --untracked-files=all > "$stage_status_after"
printf '?? REPRODUCIBILITY_METADATA.json\0' > "$preflight/expected-stage-status-after-metadata.porcelain-v1.nul"
cmp -- "$preflight/expected-stage-status-after-metadata.porcelain-v1.nul" "$stage_status_after"

metadata_sha=$(sha256sum "$stage/REPRODUCIBILITY_METADATA.json" | awk '{print $1}')
verification_tmp=$preflight/.stage-verification.$$.tmp
{
  printf 'schema=csail-gpu-visibility-stage-verification-v2\n'
  printf 'verified_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'stage_pass=true\n'
  printf 'stage=%s\n' "$stage"
  printf 'source_branch=%s\n' "$branch"
  printf 'source_head=%s\n' "$head"
  printf 'source_tree=%s\n' "$source_tree"
  printf 'lineage_base=%s\n' "$base"
  printf 'stage_git_dir=%s\n' "$stage/.git"
  printf 'stage_common_dir=%s\n' "$stage/.git"
  printf 'tracked_entries=%s\n' "$tracked_entries"
  printf 'source_tracked_manifest_sha256=%s\n' "$source_manifest_sha"
  printf 'stage_tracked_manifest_sha256=%s\n' "$stage_manifest_sha"
  printf 'source_status_sha256=%s\n' "$source_status_sha"
  printf 'source_ordinary_untracked_manifest_sha256=%s\n' "$source_untracked_sha"
  printf 'source_allowed_ignored_manifest_sha256=%s\n' "$source_allowed_ignored_sha"
  printf 'source_bundle_sha256=%s\n' "$bundle_sha"
  printf 'payload_sha256=%s\n' "$payload_sha"
  printf 'materializer_sha256=%s\n' "$helper_sha"
  printf 'metadata_path=%s\n' "$stage/REPRODUCIBILITY_METADATA.json"
  printf 'metadata_sha256=%s\n' "$metadata_sha"
  printf 'dirty_tracked_entries=0\n'
  printf 'ordinary_untracked_entries=0\n'
  printf 'allowed_ignored_input_entries=0\n'
  printf 'excluded_required_inputs=none\n'
  printf 'expected_stage_status=untracked_REPRODUCIBILITY_METADATA.json_only\n'
  printf 'bundle_verify_cwd=%s\n' "$verify_repo"
  printf 'independence_check=%s\n' "$preflight/independence-check.txt"
} > "$verification_tmp"
chmod 0600 "$verification_tmp"
mv -- "$verification_tmp" "$preflight/stage-verification.env"

sha256sum \
  "$preflight/bundle-verify.txt" \
  "$preflight/bundle-heads.txt" \
  "$preflight/stage-fsck.txt" \
  "$preflight/independence-check.txt" \
  "$source_manifest" \
  "$stage_manifest" \
  "$source_status" \
  "$source_untracked" \
  "$source_allowed_ignored" \
  "$stage_status_before" \
  "$stage_untracked_before" \
  "$stage_ignored_before" \
  "$preflight/expected-stage-status-after-metadata.porcelain-v1.nul" \
  "$stage_status_after" \
  "$stage/REPRODUCIBILITY_METADATA.json" \
  "$preflight/stage-verification.env" > "$preflight/stage-evidence.sha256"
chmod 0600 "$preflight"/*.txt "$preflight"/*.nul "$preflight"/*.env "$preflight"/*.sha256

printf 'stage_materialized=%s\n' "$stage"
printf 'stage_head=%s\n' "$head"
printf 'metadata_sha256=%s\n' "$metadata_sha"
printf 'stage_pass=true\n'
