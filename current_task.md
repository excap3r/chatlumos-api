# Task: Remove Large File from Git History

**Warning:** Rewriting Git history is destructive. Ensure collaborators are aware.

- [x] Explain history rewriting options (BFG vs. filter-branch) and ask for user preference.
- ~~[ ] Option A: BFG Repo-Cleaner~~
- [/] **Option B: git filter-branch** (Selected)
    - [x] Determine the full path of `Miniconda3-latest-MacOSX-arm64.sh` in the repository (Assumed root).
    - [x] Commit/stash unstaged changes.
    - [ ] Run filter-branch: `git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Miniconda3-latest-MacOSX-arm64.sh' --prune-empty --tag-name-filter cat -- --all`
    - [ ] Clean up references: `git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin`
    - [ ] Expire reflog: `git reflog expire --expire=now --all`
    - [ ] Garbage collect: `git gc --prune=now --aggressive`
    - [ ] Force push: `git push --force --all && git push --force --tags`
- [ ] Verify the push is successful.
- [ ] Mark all tasks as completed.
