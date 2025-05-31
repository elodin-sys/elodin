# Release

1. Ensure `CHANGELOG.md` is up to date by inspecting all changes since the last tag. You can do this with `git log $(git describe --tags --abbrev=0).. --oneline`. If there are any public-facing changes that are not in the changelog, add them to the `unreleased` section.
2. Based on the changes, decide on a patch or minor version bump. If there's a breaking API change, bump the minor version. Since we're in pre-1.0, we can make breaking changes in minor versions.
3. Add a new section to `CHANGELOG.md` with the new version number and move all changes from the `unreleased` section to the new section. If there are no changes in the `unreleased` section, you can skip this step and the next step.
4. Update the version in `Cargo.toml` to the new version number. Make sure to update the version numbers in the `[workspace.dependencies]` section as well. Run `cargo check` to ensure that the lock file is updated.
5. Run `just public-changelog` to generate the public-facing changelog in the docs site from the `CHANGELOG.md` file.
6. Create a new release branch with the new version number. For example, if the new version is `0.1.2`, create a branch named `release/v0.1.2` by running `git checkout -b release/v0.1.2`.
7. Add and commit the changes to the release branch by running `git commit -am "chore: release v0.1.2"`.
8. Push the release branch to the remote repository by running `git push -u origin release/v0.1.2`.
9. Create a pull request from the release branch to the `main` branch. If using the GitHub CLI, you can run `gh pr create --base main --head release/v0.1.2 --title "Release v0.1.2"` to create the pull request.
10. Once the pull request is merged and the CI pipeline has passed for the `main` branch, run `just auto-tag` on the updated `main` branch. This recipe will:
    - Create a new git tag with the new version number.
    - Push the tag to the remote repository, which will trigger GitHub Actions to start building release artifacts.
    - Re-tag the latest container images with the new version number.
11. Once the release artifacts are built successfully, a new release will be published to GitHub at https://github.com/elodin-sys/elodin/releases. After the release is published, run `just promote v0.1.2` to promote the artifacts from GitHub to S3 and PyPi. This is what enables end-users to download and install both the new CLI and Python packages. You may want to do some local testing of the release artifacts before promoting them to production.
12. After the release artifacts are promoted, rollout the new container images (including the docs site) to production by running `just release v0.1.2`. This will deploy any changes to the Kubernetes cluster into production. Ensure nothing major is broken by checking https://app.elodin.systems and https://docs.elodin.systems.
13. Sync the open-source repository with the changes in the private repository by running `just force-push-open-source`. This will update the open-source repository with the latest changes from the private repository.
