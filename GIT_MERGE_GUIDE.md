# Git Merge Conflict Resolution Guide

## ✅ Configuration Applied

I've configured your repository to use **merge** instead of **rebase** when pulling:

```bash
git config pull.rebase false
git config merge.conflictstyle diff3
```

This means future `git pull` commands will create merge commits instead of trying to rebase, which is easier to handle.

---

## 🚀 Quick Commands

### Option 1: Use the Helper Script (Recommended)
```bash
./git-pull-helper.sh
```

### Option 2: Manual Pull
```bash
git pull origin feature/rebuild-app
```

---

## ⚠️ If You Get Merge Conflicts

When you see conflicts, you'll see files marked like this:

```
<<<<<<< HEAD
Your local changes
=======
Remote changes
>>>>>>> commit-hash
```

### Strategy 1: Keep YOUR Local Changes
```bash
# For each conflicted file:
git checkout --ours backend/dish_processing.py
git checkout --ours backend/routes/chat.py
git add backend/dish_processing.py backend/routes/chat.py
git commit -m "Resolved merge conflicts - kept local changes"
```

### Strategy 2: Keep THEIR Remote Changes
```bash
# For each conflicted file:
git checkout --theirs backend/dish_processing.py
git checkout --theirs backend/routes/chat.py
git add backend/dish_processing.py backend/routes/chat.py
git commit -m "Resolved merge conflicts - accepted remote changes"
```

### Strategy 3: Manual Resolution
1. Open the conflicted file in your editor
2. Look for conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
3. Edit the file to keep what you want
4. Remove the conflict markers
5. Save the file
6. Run:
```bash
git add <file>
git commit -m "Resolved merge conflicts manually"
```

---

## 🔧 Useful Commands

### Check current status
```bash
git status
```

### See which files have conflicts
```bash
git diff --name-only --diff-filter=U
```

### Abort a merge (start over)
```bash
git merge --abort
```

### Abort a rebase (if you accidentally started one)
```bash
git rebase --abort
```

### View the conflict in detail
```bash
git diff <filename>
```

---

## 📝 Best Practices

1. **Before pulling**, commit or stash your local changes:
   ```bash
   git add .
   git commit -m "Your commit message"
   # OR
   git stash
   ```

2. **Pull regularly** to avoid large conflicts:
   ```bash
   git pull origin feature/rebuild-app
   ```

3. **Communicate with team** about major changes to avoid conflicts

4. **Use feature branches** for experimental work

---

## 🆘 Emergency Commands

### If everything is messed up and you want to reset to remote:
```bash
# WARNING: This will DELETE all your local changes!
git fetch origin
git reset --hard origin/feature/rebuild-app
```

### If you want to save your local changes first:
```bash
# Save your work
git stash save "My local changes backup"

# Reset to remote
git fetch origin
git reset --hard origin/feature/rebuild-app

# Optionally restore your changes
git stash pop
```

---

## 📚 Current Situation

Your local branch: `feature/rebuild-app`
Remote branch: `origin/feature/rebuild-app`

Remote has 4 commits ahead of you:
- Add unit tests for chat, extraction, and location matching functionalities
- Enhance dish filtering and recommendation logic
- Enhance dish processing and recommendation logic
- Enhance dish extraction and recommendation logic

**Recommendation**: Pull the remote changes and resolve conflicts by keeping your local changes (since we just made improvements).

