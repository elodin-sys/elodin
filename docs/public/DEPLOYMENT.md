# Documentation Deployment Guide

This guide explains how the Elodin documentation site is built and deployed to GitHub Pages.

## 🚀 Overview

Our documentation is:
- **Built with:** [Zola](https://www.getzola.org/) static site generator
- **Hosted on:** GitHub Pages
- **Domain:** https://docs.elodin.systems
- **Auto-deployed:** On every push to `main` branch
- **PR Previews:** Automatic preview deployments for pull requests

## 📁 Project Structure

```
docs/public/
├── config.toml        # Zola configuration
├── content/          # Markdown content files
├── static/           # Static assets (images, videos, etc.)
├── sass/            # SCSS stylesheets
├── templates/       # HTML templates
└── public/          # Built site (git-ignored)
```

## 🔧 Local Development

### Prerequisites
- Install Zola: `brew install zola` (macOS) or see [installation guide](https://www.getzola.org/documentation/getting-started/installation/)

### Running Locally
```bash
cd docs/public
zola serve
# Visit http://127.0.0.1:1111
```

### Building Locally
```bash
cd docs/public
zola build
# Output will be in docs/public/public/
```

## 🚢 Deployment Process

### Automatic Production Deployment
The site automatically deploys to production when:
1. Changes are pushed to `main` branch
2. Changes affect files in `docs/public/` directory
3. GitHub Actions workflow completes successfully
4. Site updates at https://docs.elodin.systems

### Pull Request Previews
Every pull request automatically gets a preview deployment:
1. Open a PR with changes to `docs/public/`
2. GitHub Actions builds the preview
3. Preview deploys to: `https://docs.elodin.systems/preview/pr-{number}/`
4. A comment is posted on the PR with the preview link
5. Preview updates automatically when you push new commits
6. Preview is removed when PR is closed or merged

### Manual Deployment
You can manually trigger deployment:
1. Go to [Actions tab](../../actions)
2. Select "Deploy Documentation" workflow
3. Click "Run workflow"
4. Select `main` branch and click "Run workflow"

### How It Works

The deployment is handled by `.github/workflows/deploy-docs.yml`:

#### Production Deployment (main branch):
1. **Build Stage:**
   - Checks out the repository
   - Installs Zola v0.19.2
   - Runs `zola build` in `docs/public/`
   - Deploys to root of `gh-pages` branch

2. **Verify Stage:**
   - Confirms deployment success
   - Performs health check on https://docs.elodin.systems

#### PR Preview Deployment:
1. **Build Stage:**
   - Checks out the PR code
   - Builds with preview-specific base URL
   - Deploys to `gh-pages` branch under `/preview/pr-{number}/`

2. **Comment Stage:**
   - Posts/updates comment on PR with preview link
   - Includes deployment timestamp

3. **Cleanup Stage (when PR closes):**
   - Removes preview directory from `gh-pages` branch
   - Comments on PR confirming removal

## 📝 Making Changes

### Content Updates
1. Edit markdown files in `docs/public/content/`
2. Test locally with `zola serve`
3. Commit and push to `main`
4. Deployment happens automatically

### Style Changes
1. Edit SCSS files in `docs/public/sass/`
2. Zola automatically compiles SCSS → CSS
3. Test locally before pushing

### Adding Assets
1. Place files in `docs/public/static/`
2. Reference from content as `/path/to/asset`
3. Keep file sizes reasonable (videos are already optimized)

## 🎥 Video Encoding

For adding new videos, use the provided justfile:
```bash
cd docs/public
just encode path/to/video.mov
```
This creates optimized H.264 and AV1 versions.

## 🔍 Troubleshooting

### Build Failures
- Check [GitHub Actions](../../actions) for error logs
- Ensure Zola version matches (0.19.2)
- Validate `config.toml` syntax
- Check for broken internal links

### Domain Issues
- DNS is managed through Cloudflare
- GitHub Pages custom domain is set in repository settings
- CNAME file is automatically handled by GitHub

### Preview Deployment Issues
- Ensure PR has write permissions (from repo collaborators)
- Check that `gh-pages` branch exists
- Verify GitHub Pages is set to deploy from `gh-pages` branch
- Preview URLs only work after DNS is properly configured

### Local vs Production Differences
- Base URL is set to `https://docs.elodin.systems` in config.toml
- Local development uses `http://127.0.0.1:1111`
- Zola handles this automatically

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes in `docs/public/`
4. Test locally with `zola serve`
5. Submit a pull request
6. **Automatic preview deployment:**
   - Wait ~2 minutes for the preview to build
   - Find the preview URL in the automated PR comment
   - Preview URL format: `https://docs.elodin.systems/preview/pr-{number}/`
   - Preview updates automatically with each push
   - Share preview link with reviewers for feedback

## 📊 Monitoring

- View deployment status: [GitHub Actions](../../actions)
- Check site health: https://docs.elodin.systems
- GitHub Pages status: [Repository Settings → Pages](../../settings/pages)

## 💡 Tips

- Keep content in markdown for easy editing
- Use relative links between docs pages
- Optimize images before adding (use WebP when possible)
- Test responsive design locally
- Check search functionality after major changes

## 🆘 Getting Help

- Check this guide first
- Review [Zola documentation](https://www.getzola.org/documentation/)
- Open an issue for documentation problems
- Discuss in project Discord/Slack

---

*Last updated: December 2024*
