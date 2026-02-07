#!/bin/bash
# Release script for Muxide
# Usage: ./scripts/release.sh <version>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.3"
    exit 1
fi

VERSION=$1

echo "🚀 Preparing release $VERSION"

# Update version in Cargo.toml
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml

# Update CHANGELOG.md (you'll need to add the entry manually)
echo "📝 Please update CHANGELOG.md with the new version notes"
echo "   Then press Enter to continue..."
read

# Commit version bump
git add Cargo.toml CHANGELOG.md
git commit -m "Release $VERSION"

# Create and push tag
git tag "v$VERSION"
git push origin main
git push origin "v$VERSION"

echo "✅ Release $VERSION tagged and pushed!"
echo "   GitHub Actions will now:"
echo "   - Build and publish binaries to the release"
echo "   - Publish to crates.io"
echo ""
echo "   Check the Actions tab for progress."