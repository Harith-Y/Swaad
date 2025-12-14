#!/bin/bash

# Git Pull Helper Script
# This script helps you pull changes from remote and handle merge conflicts

echo "🔄 Fetching latest changes from remote..."
git fetch origin

echo ""
echo "📊 Checking for differences between local and remote..."
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})
BASE=$(git merge-base @ @{u})

if [ $LOCAL = $REMOTE ]; then
    echo "✅ Already up to date!"
    exit 0
elif [ $LOCAL = $BASE ]; then
    echo "⬇️  Remote has new changes. Pulling..."
    git pull origin $(git branch --show-current)
elif [ $REMOTE = $BASE ]; then
    echo "⬆️  You have local changes that need to be pushed."
    echo "Run: git push origin $(git branch --show-current)"
    exit 0
else
    echo "🔀 Both local and remote have changes. Merging..."
    
    # Try to pull with merge strategy
    git pull origin $(git branch --show-current)
    
    # Check if there are conflicts
    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠️  Merge conflicts detected!"
        echo ""
        echo "Conflicted files:"
        git diff --name-only --diff-filter=U
        echo ""
        echo "Options:"
        echo "1. Keep YOUR changes (local):  git checkout --ours <file> && git add <file>"
        echo "2. Keep THEIR changes (remote): git checkout --theirs <file> && git add <file>"
        echo "3. Manually resolve: Edit the files and run 'git add <file>'"
        echo ""
        echo "After resolving all conflicts, run: git commit"
        exit 1
    else
        echo "✅ Merge successful!"
    fi
fi

