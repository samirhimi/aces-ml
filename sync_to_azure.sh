#!/bin/bash

GITHUB_REPO_DIR="/home/devops/aces-ml"
AZURE_REPO="https://$AZURE_PAT@dev.azure.com/dev-org/aces-ml/_git/aces-ml"

cd "$GITHUB_REPO_DIR" || exit 1

# Pull latest changes from GitHub
git pull origin main

# Push to Azure DevOps
git push "$AZURE_REPO" main
