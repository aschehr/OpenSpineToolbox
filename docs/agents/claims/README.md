# Claim files

A claim file is the first commit of a claim: `docs/agents/claims/<issue number>.md`, added by the first commit on the branch `claude/<issue number>`, which is pushed from the default branch as docs/agents/PROTOCOL.md says. It names the worker, the plan and the files the work expects to touch, so the board can show who holds what and which open work meets in the same files, and so a released claim leaves its plan behind for the next worker: the coordinator's release comment quotes the plan, and the branch `released/<issue number>-<sha>` the board renames it to keeps the file. The worker's name is its label without the `worker:` prefix (`nightly` for the label worker:nightly): a person's sessions take the person's GitHub login, unless the session's environment sets SYMPHONY_WORKER to another, and a routine or a worker in Actions the name its file under .github/symphony/workers/ gives it (docs/agents/WORKERS.md). The board reads it from the `- Worker:` line, or from the older `- Account:` line of a claim made before workers had one name, and takes the list under "## Files to touch" for its Files column and its Overlaps until the pull request has files of its own. The pull request that finishes the issue deletes the file. Files here are named by issue number only; this README is the one exception, and nothing reads it as a claim.

The template, with N the issue's number and NAME the worker's name:

```
# Claim: #N

- Issue: https://github.com/aschehr/OpenSpineToolbox/issues/N
- Worker: NAME
- Branch: claude/N

## Plan

Two to five sentences: what will change and why, in the order it will land.

## Files to touch

- the paths this work expects to change, one per line
```

Keep it short; the pull request carries the detail. A claim is not a design document.
