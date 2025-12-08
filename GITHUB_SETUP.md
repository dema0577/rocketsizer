# GitHub Setup Instructions

This guide will help you publish RocketSizer on GitHub.

## Initial Setup

1. **Create a new repository on GitHub**
   - Go to https://github.com/new
   - Name it `rocket-sizer` (or your preferred name)
   - Choose Public or Private
   - **Do NOT** initialize with README, .gitignore, or license (we already have these)

2. **Initialize Git in your local project**
   ```bash
   cd "/Users/matteodema77gmail.com/Desktop/rocket sizer"
   git init
   ```

3. **Add all files**
   ```bash
   git add .
   ```

4. **Make your first commit**
   ```bash
   git commit -m "Initial commit: RocketSizer educational rocket engine sizing tool"
   ```

5. **Connect to GitHub repository**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/rocket-sizer.git
   # Replace YOUR_USERNAME with your GitHub username
   ```

6. **Push to GitHub**
   ```bash
   git branch -M main
   git push -u origin main
   ```

## Customize Before Publishing

Before pushing, make sure to:

1. **Update README.md**
   - Replace `yourusername` with your actual GitHub username
   - Replace "Your Name" with your actual name
   - Add any additional information you want

2. **Update LICENSE**
   - Replace the copyright year and name if needed

3. **Optional: Add screenshots**
   - Take screenshots of your application
   - Add them to a `docs/` or `images/` folder
   - Reference them in the README

## Repository Settings (Recommended)

After pushing to GitHub:

1. Go to your repository settings
2. Enable "Issues" (Settings → Features)
3. Enable "Discussions" (optional, for community interaction)
4. Add repository topics: `rocket`, `propulsion`, `aerospace`, `streamlit`, `python`, `educational`
5. Add a repository description: "Educational software for sizing liquid propellant rocket engines"

## Optional: GitHub Pages

If you want to host the app documentation:
1. Go to Settings → Pages
2. Select branch `main` and folder `/docs` (if you create one)
3. Or use GitHub Actions to deploy Streamlit (requires additional setup)

## After Publishing

- Share your repository with others!
- Consider adding a demo link if you host it somewhere
- Engage with issues and pull requests
- Keep the documentation updated

---

Good luck with your repository! 🚀

