# Git Best Practices for Enterprise Development

> **Purpose:** Establish professional Git workflows and practices used in enterprise software development.  
> **Applies to:** All projects in this learning path and future professional work.  
> **Last Updated:** February 27, 2026

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Essential Commands](#essential-commands)
3. [Branching Strategy](#branching-strategy)
4. [Commit Conventions](#commit-conventions)
5. [Pull Request Workflow](#pull-request-workflow)
6. [Code Review Practices](#code-review-practices)
7. [Handling Conflicts](#handling-conflicts)
8. [Git Configuration](#git-configuration)
9. [Security Best Practices](#security-best-practices)
10. [GitHub Integration](#github-integration)
11. [Common Scenarios](#common-scenarios)
12. [Troubleshooting](#troubleshooting)

---

## Core Concepts

### The Three States of Git

```
┌─────────────┐     git add      ┌─────────────┐    git commit    ┌─────────────┐
│  Working    │ ───────────────► │   Staging   │ ────────────────►│    Local    │
│  Directory  │                  │    Area     │                  │  Repository │
└─────────────┘                  └─────────────┘                  └─────────────┘
       │                                                                 │
       │                         git checkout                            │
       │◄────────────────────────────────────────────────────────────────│
       │                                                                 │
       │                          git push                               │
       │                                                                 ▼
       │                                                          ┌─────────────┐
       │◄──────────────────── git pull ───────────────────────────│   Remote    │
       │                                                          │  (GitHub)   │
                                                                  └─────────────┘
```

### Key Terminology

| Term | Definition |
|------|------------|
| **Repository (repo)** | A directory tracked by Git containing all project files and history |
| **Commit** | A snapshot of your changes with a unique identifier (SHA) |
| **Branch** | An independent line of development |
| **HEAD** | Pointer to the current branch/commit you're working on |
| **Remote** | A repository hosted elsewhere (e.g., GitHub) |
| **Origin** | Default name for the primary remote repository |
| **Staging Area** | Intermediate area where commits are prepared |
| **Merge** | Combining changes from one branch into another |
| **Rebase** | Reapplying commits on top of another branch |
| **Pull Request (PR)** | Request to merge changes with code review |

---

## Essential Commands

### Daily Workflow Commands

```bash
# Check status of your working directory
git status

# See what changes you've made
git diff                    # Unstaged changes
git diff --staged           # Staged changes

# Stage changes
git add <file>              # Stage specific file
git add .                   # Stage all changes
git add -p                  # Interactive staging (review each change)

# Commit changes
git commit -m "message"     # Commit with message
git commit                  # Opens editor for detailed message
git commit --amend          # Modify last commit (before push only!)

# Push changes
git push                    # Push to tracked branch
git push -u origin <branch> # Push and set upstream tracking

# Pull changes
git pull                    # Fetch and merge
git pull --rebase           # Fetch and rebase (cleaner history)

# View history
git log                     # Full commit history
git log --oneline           # Compact history
git log --graph --oneline   # Visual branch history
git log -p                  # History with diffs
```

### Branch Management

```bash
# List branches
git branch                  # Local branches
git branch -r               # Remote branches
git branch -a               # All branches

# Create and switch branches
git branch <name>           # Create branch
git checkout <name>         # Switch to branch
git checkout -b <name>      # Create and switch (shortcut)
git switch <name>           # Modern way to switch (Git 2.23+)
git switch -c <name>        # Create and switch (Git 2.23+)

# Delete branches
git branch -d <name>        # Delete merged branch
git branch -D <name>        # Force delete unmerged branch
git push origin --delete <name>  # Delete remote branch

# Rename branch
git branch -m <old> <new>   # Rename local branch
```

### Synchronization

```bash
# Fetch updates (doesn't modify working directory)
git fetch                   # Fetch all remotes
git fetch origin            # Fetch specific remote

# Merge changes
git merge <branch>          # Merge branch into current
git merge --no-ff <branch>  # Merge with merge commit (preserves history)

# Rebase
git rebase <branch>         # Rebase current branch onto another
git rebase -i HEAD~3        # Interactive rebase last 3 commits

# Sync with remote
git pull origin main        # Pull specific branch
git push origin main        # Push specific branch
```

### Undoing Changes

```bash
# Discard working directory changes
git checkout -- <file>      # Discard changes to file
git restore <file>          # Modern way (Git 2.23+)

# Unstage changes
git reset HEAD <file>       # Unstage file
git restore --staged <file> # Modern way (Git 2.23+)

# Reset commits (USE WITH CAUTION)
git reset --soft HEAD~1     # Undo commit, keep changes staged
git reset --mixed HEAD~1    # Undo commit, keep changes unstaged (default)
git reset --hard HEAD~1     # Undo commit, discard changes (DANGEROUS)

# Revert a commit (safe, creates new commit)
git revert <commit-sha>     # Create commit that undoes changes

# Recover deleted commits
git reflog                  # View all HEAD movements
git checkout <sha>          # Recover to specific point
```

### Stashing

```bash
# Temporarily save changes
git stash                   # Stash changes
git stash save "message"    # Stash with description
git stash -u                # Include untracked files

# Retrieve stashed changes
git stash list              # List all stashes
git stash pop               # Apply and remove latest stash
git stash apply             # Apply but keep stash
git stash apply stash@{2}   # Apply specific stash

# Clean up stashes
git stash drop              # Remove latest stash
git stash clear             # Remove all stashes
```

---

## Branching Strategy

### Git Flow (Recommended for Enterprise)

```
main (production)
  │
  ├── develop (integration)
  │     │
  │     ├── feature/prompt-template-system
  │     │     └── (feature work)
  │     │
  │     ├── feature/evaluation-framework
  │     │     └── (feature work)
  │     │
  │     └── bugfix/fix-token-counting
  │           └── (bug fix work)
  │
  ├── release/v1.0.0
  │     └── (release preparation)
  │
  └── hotfix/critical-security-fix
        └── (emergency fixes)
```

### Branch Naming Conventions

```bash
# Feature branches
feature/add-prompt-versioning
feature/implement-rag-pipeline
feature/JIRA-123-user-authentication

# Bug fix branches
bugfix/fix-memory-leak
bugfix/correct-token-calculation

# Hotfix branches (production emergencies)
hotfix/critical-security-patch
hotfix/fix-api-timeout

# Release branches
release/v1.0.0
release/v2.1.0

# Experiment branches
experiment/test-new-embedding-model
spike/evaluate-langgraph
```

### Branch Protection Rules (GitHub)

Configure these on your main branches:

| Rule | Purpose |
|------|---------|
| Require pull request reviews | Ensure code review before merge |
| Require status checks to pass | Ensure CI/CD passes |
| Require branches to be up to date | Prevent merge conflicts |
| Include administrators | Apply rules to everyone |
| Restrict pushes | Prevent direct commits to main |

---

## Commit Conventions

### Conventional Commits Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(retrieval): add hybrid search capability` |
| `fix` | Bug fix | `fix(agent): resolve memory leak in conversation history` |
| `docs` | Documentation | `docs(readme): add installation instructions` |
| `style` | Formatting, no code change | `style: fix indentation in config module` |
| `refactor` | Code restructuring | `refactor(prompts): extract template loading logic` |
| `perf` | Performance improvement | `perf(embeddings): cache frequently used embeddings` |
| `test` | Adding/updating tests | `test(evaluation): add unit tests for metrics` |
| `chore` | Maintenance tasks | `chore: update dependencies` |
| `ci` | CI/CD changes | `ci: add GitHub Actions workflow` |
| `build` | Build system changes | `build: configure Docker multi-stage build` |
| `revert` | Reverting changes | `revert: undo prompt template changes` |

### Good Commit Messages

```bash
# ✅ Good: Clear, specific, imperative mood
git commit -m "feat(rag): implement document chunking with semantic boundaries"
git commit -m "fix(agent): handle empty tool response gracefully"
git commit -m "docs(phase1): add evaluation framework documentation"

# ❌ Bad: Vague, past tense, no context
git commit -m "fixed stuff"
git commit -m "updates"
git commit -m "WIP"
```

### Multi-line Commit Messages

```bash
git commit -m "feat(evaluation): add LLM-as-judge evaluation

Implement evaluation framework using GPT-4 as judge for:
- Response relevance scoring
- Factual accuracy checking  
- Tone appropriateness

Closes #42"
```

### Commit Best Practices

1. **Atomic commits**: One logical change per commit
2. **Commit often**: Small, frequent commits are easier to review
3. **Test before committing**: Ensure code works
4. **Never commit secrets**: Use environment variables
5. **Don't commit generated files**: Use .gitignore
6. **Write meaningful messages**: Future you will thank you

---

## Pull Request Workflow

### Creating a Pull Request

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: implement feature"
   ```

3. **Push to remote**
   ```bash
   git push -u origin feature/my-feature
   ```

4. **Create PR on GitHub**
   - Go to repository
   - Click "Compare & pull request"
   - Fill in PR template

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] Dependent changes merged

## Screenshots (if applicable)

## Related Issues
Closes #123
```

### PR Best Practices

| Practice | Why |
|----------|-----|
| Keep PRs small (<400 lines) | Easier to review, faster feedback |
| One concern per PR | Clear scope, easier rollback |
| Include context | Help reviewers understand |
| Respond to feedback promptly | Keep momentum |
| Don't force push after review | Preserves discussion context |
| Squash commits if messy | Clean history on merge |

---

## Code Review Practices

### As a Reviewer

```markdown
## Review Checklist

### Correctness
- [ ] Does the code do what it's supposed to?
- [ ] Are edge cases handled?
- [ ] Are there any bugs?

### Design
- [ ] Is the code well-structured?
- [ ] Does it follow project patterns?
- [ ] Is there unnecessary complexity?

### Readability
- [ ] Is the code easy to understand?
- [ ] Are names meaningful?
- [ ] Are comments helpful (not obvious)?

### Testing
- [ ] Are there adequate tests?
- [ ] Do tests cover edge cases?
- [ ] Are tests readable?

### Security
- [ ] No hardcoded secrets?
- [ ] Input validation present?
- [ ] No obvious vulnerabilities?

### Performance
- [ ] No obvious performance issues?
- [ ] Appropriate data structures used?
```

### Review Comment Conventions

```markdown
# Prefix comments with intent

[blocking] This will cause a null pointer exception
[suggestion] Consider using a dictionary here for O(1) lookup
[question] Why did you choose this approach over X?
[nit] Typo in variable name
[praise] Great solution for handling the edge case!
```

### As an Author

1. **Self-review first**: Review your own code before requesting
2. **Provide context**: Explain why, not just what
3. **Be responsive**: Address feedback promptly
4. **Don't take it personally**: Reviews improve code quality
5. **Ask questions**: If feedback is unclear, ask

---

## Handling Conflicts

### Understanding Merge Conflicts

```
<<<<<<< HEAD
Your changes (current branch)
=======
Their changes (incoming branch)
>>>>>>> feature-branch
```

### Resolving Conflicts

```bash
# 1. Start merge
git merge feature-branch

# 2. See conflicted files
git status

# 3. Open conflicted file and resolve manually
# Remove conflict markers and choose/combine changes

# 4. Mark as resolved
git add <resolved-file>

# 5. Complete merge
git commit
```

### Conflict Prevention

1. **Pull frequently**: Stay up to date with main
2. **Small PRs**: Less chance of conflicts
3. **Communicate**: Coordinate with team on shared files
4. **Use feature flags**: Merge incomplete features safely

### Tools for Conflict Resolution

```bash
# Use visual merge tool
git mergetool

# Abort merge if needed
git merge --abort

# See conflict details
git diff --name-only --diff-filter=U
```

---

## Git Configuration

### Essential Configuration

```bash
# Identity (use personal for this repo)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Default branch name
git config --global init.defaultBranch main

# Default editor
git config --global core.editor "code --wait"  # VS Code
git config --global core.editor "vim"          # Vim

# Line endings (important for cross-platform)
git config --global core.autocrlf input  # macOS/Linux
git config --global core.autocrlf true   # Windows

# Push behavior
git config --global push.default current

# Pull behavior
git config --global pull.rebase true

# Colorful output
git config --global color.ui auto
```

### Useful Aliases

```bash
# Add to ~/.gitconfig or run git config --global alias.<name> "<command>"

[alias]
    # Shortcuts
    co = checkout
    br = branch
    ci = commit
    st = status
    
    # Logging
    lg = log --oneline --graph --decorate
    hist = log --pretty=format:'%h %ad | %s%d [%an]' --graph --date=short
    
    # Undo
    unstage = reset HEAD --
    undo = reset --soft HEAD~1
    
    # Branch management
    branches = branch -a
    remotes = remote -v
    
    # Show last commit
    last = log -1 HEAD --stat
    
    # Diff
    df = diff --word-diff
    
    # Clean merged branches
    cleanup = "!git branch --merged | grep -v '\\*\\|main\\|develop' | xargs -n 1 git branch -d"
```

### .gitignore Best Practices

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
venv/
.env
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
.cache/
secrets.json
*.local

# AI/ML specific
*.h5
*.pkl
*.model
mlruns/
wandb/

# Never commit
.env
.env.local
*.pem
*.key
credentials.json
```

---

## Security Best Practices

### Never Commit Secrets

```bash
# ❌ NEVER do this
API_KEY = "sk-1234567890abcdef"
password = "mysecretpassword"

# ✅ Use environment variables
API_KEY = os.environ.get("OPENAI_API_KEY")
```

### If You Accidentally Commit Secrets

```bash
# 1. Remove from history (if not pushed)
git reset --soft HEAD~1
# Remove secret from file
git add .
git commit -m "feat: add feature without secrets"

# 2. If already pushed - ROTATE THE SECRET IMMEDIATELY
# Then use BFG Repo-Cleaner or git filter-branch
# Consider the secret compromised regardless
```

### Signed Commits

```bash
# Configure GPG signing
git config --global user.signingkey <your-key-id>
git config --global commit.gpgsign true

# Sign a commit
git commit -S -m "Signed commit message"

# Verify signatures
git log --show-signature
```

### Repository Security

1. **Enable branch protection** on main branches
2. **Require signed commits** for sensitive repos
3. **Use GitHub security features**: Dependabot, secret scanning
4. **Audit access** regularly
5. **Use SSH keys** over HTTPS tokens when possible

---

## GitHub Integration

### GitHub Features to Use

| Feature | Purpose |
|---------|---------|
| **Issues** | Track bugs, features, tasks |
| **Projects** | Kanban-style project management |
| **Actions** | CI/CD automation |
| **Discussions** | Team communication |
| **Wiki** | Documentation |
| **Releases** | Version management |
| **Security** | Vulnerability scanning |

### GitHub Actions (CI/CD)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
          
      - name: Run tests
        run: pytest
        
      - name: Run linting
        run: |
          pip install ruff
          ruff check .
```

### Issue and PR Templates

Create `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug Report
about: Report a bug
---

## Bug Description
A clear description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: 
- Python version:
- Package versions:
```

---

## Common Scenarios

### Starting a New Feature

```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/my-new-feature

# 3. Work on feature (commit regularly)
git add .
git commit -m "feat: implement part 1"
# ... more commits ...

# 4. Stay up to date with main
git fetch origin
git rebase origin/main

# 5. Push and create PR
git push -u origin feature/my-new-feature
```

### Updating Your Branch with Main

```bash
# Option 1: Rebase (cleaner history)
git fetch origin
git rebase origin/main

# Option 2: Merge (preserves history)
git fetch origin
git merge origin/main
```

### Fixing a Bug in Production

```bash
# 1. Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-bug

# 2. Fix and commit
git add .
git commit -m "fix: resolve critical bug"

# 3. Push and create PR directly to main
git push -u origin hotfix/critical-bug
# Create PR, get quick review, merge

# 4. Also merge to develop
git checkout develop
git merge hotfix/critical-bug
git push origin develop
```

### Squashing Commits Before Merge

```bash
# Interactive rebase to squash last 5 commits
git rebase -i HEAD~5

# In editor, change 'pick' to 'squash' for commits to combine
pick abc1234 First commit
squash def5678 Second commit
squash ghi9012 Third commit
# Save and edit combined commit message
```

### Cherry-Picking a Commit

```bash
# Apply specific commit to current branch
git cherry-pick <commit-sha>

# Cherry-pick without committing
git cherry-pick -n <commit-sha>
```

---

## Troubleshooting

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| "Permission denied" on push | Check SSH keys or token permissions |
| "Diverged branches" | `git pull --rebase` then `git push` |
| Accidentally committed to main | `git reset HEAD~1` then create branch |
| Need to undo last commit | `git reset --soft HEAD~1` |
| Lost commits after reset | `git reflog` to find and recover |
| Large file rejected | Use Git LFS or remove from history |

### Recovering Lost Work

```bash
# Find lost commits
git reflog

# Restore to specific point
git checkout <sha>

# Create branch from recovered state
git checkout -b recovered-work
```

### Cleaning Up

```bash
# Remove untracked files (preview)
git clean -n

# Remove untracked files (execute)
git clean -f

# Remove untracked files and directories
git clean -fd

# Remove ignored files too
git clean -fdx
```

---

## Quick Reference Card

### Daily Commands
```bash
git status              # What's changed?
git add .               # Stage all changes
git commit -m "msg"     # Commit changes
git push                # Push to remote
git pull                # Get latest changes
```

### Branch Commands
```bash
git checkout -b name    # Create and switch branch
git checkout main       # Switch to main
git merge branch        # Merge branch into current
git branch -d name      # Delete merged branch
```

### History Commands
```bash
git log --oneline       # View history
git diff                # See changes
git blame file          # Who changed what
git show sha            # View specific commit
```

### Undo Commands
```bash
git checkout -- file    # Discard file changes
git reset HEAD file     # Unstage file
git reset --soft HEAD~1 # Undo last commit (keep changes)
git revert sha          # Undo commit (new commit)
```

---

## Additional Resources

- [Pro Git Book](https://git-scm.com/book/en/v2) — Free, comprehensive guide
- [GitHub Docs](https://docs.github.com) — Official GitHub documentation
- [Conventional Commits](https://www.conventionalcommits.org) — Commit message standard
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/) — Branching model
- [Learn Git Branching](https://learngitbranching.js.org) — Interactive tutorial

---

*This document covers enterprise-level Git practices. Apply these consistently across all projects.*
