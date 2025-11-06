# Release

1. Ensure `CHANGELOG.md` is up to date by inspecting all changes since the last tag. You can do this with `git log $(git describe --tags --abbrev=0).. --oneline`. If there are any public-facing changes that are not in the changelog, add them to the `unreleased` section.
2. Based on the changes, decide on a patch or minor version bump. If there's a breaking API change, bump the minor version. Since we're in pre-1.0, we can make breaking changes in minor versions.
3. Add a new section to `CHANGELOG.md` with the new version number and move all changes from the `unreleased` section to the new section. If there are no changes in the `unreleased` section, you can skip this step and the next step.
4. Update the version in `Cargo.toml` to the new version number, and update `docs/public/config.toml`. Make sure to update the version numbers in the `[workspace.dependencies]` section as well. Run `cargo check` to ensure that the lock file is updated.
5. Run `just public-changelog` to generate the public-facing changelog in the docs site from the `CHANGELOG.md` file.
6. Choose your next version number. In preparation for running the following commands, set the `VERSION` environment variable. For example if the new version is `0.1.2`, then run `export VERSION=0.1.2`.
7. Create a new release branch with the new version number. Create a branch named `release/v$VERSION` by running `git checkout -b release/v$VERSION`.
8. Add and commit the changes to the release branch by running `git commit -m "chore: release v$VERSION"`.
9. Push the release branch to the remote repository by running `git push -u origin release/v$VERSION`.
10. Create a pull request from the release branch to the `main` branch. If using the GitHub CLI, you can run `gh pr create --base main --head release/v$VERSION --title "Release v$VERSION"` to create the pull request.
11. Once the pull request is merged and the CI pipeline has passed for the `main` branch, run `just tag v$VERSION` on the updated `main` branch. This recipe will:
    - Create a new git tag with the new version number.
    - Push the tag to the remote repository, which will trigger GitHub Actions to start building release artifacts.
    - Re-tag the latest container images with the new version number.
12. Once the release artifacts are built successfully, a new release will be published to GitHub at https://github.com/elodin-sys/elodin/releases. After the release is published, run `just promote v$VERSION` to promote the artifacts from GitHub to S3 and PyPi. This is what enables end-users to download and install both the new CLI and Python packages. You may want to do some local testing of the release artifacts before promoting them to production.
13. Once released, we want to bump the version as an alpha so that what was just released and what's on the 'main' branch will not be easily confused. If the version released was `0.1.2` then the next version would be `0.1.3-alpha.0`. We can do this with the [semver-cli](https://crates.io/crates/semver-cli) tool. 
```sh
export NEXT_VERSION="$(semver-cli $VERSION --increment)-alpha.0"
echo $NEXT_VERSION

git checkout -b chore/v$NEXT_VERSION
```
14. Update the version in `Cargo.toml` to the new version number and run `cargo check`.
15. Commit, push, and create PR.
```sh
git commit -am "chore: Bump to v$NEXT_VERSION"
git push -u origin chore/v$NEXT_VERSION
gh pr create --base main --head chore/v$NEXT_VERSION --title "Bump to v$NEXT_VERSION"
```
16. Review and merge PR.
