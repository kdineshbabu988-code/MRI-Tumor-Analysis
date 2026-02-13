# 🚀 Procedure: Pushing to GitHub

Follow these steps to upload your Brain Tumor Classification project to a GitHub repository.

## 1. Prepare your Repository
Before running commands, create a new repository on GitHub:
1. Go to [github.com/new](https://github.com/new).
2. Name it (e.g., `MRI-Brain-Tumor-Classification`).
3. Keep it **Public** or **Private** as per your preference.
4. **Do NOT** initialize with a README, license, or gitignore (we already have them).
5. Click **Create repository**.

## 2. Initialize Git Locally
Open your terminal in the project folder and run:

```bash
# Initialize the local directory as a Git repository
git init

# Add all files (the .gitignore will automatically skip large data)
git add .

# Commit the changes
git commit -m "Initial commit: Production-ready MRI classification pipeline"
```

## 3. Link to GitHub
Copy the URL of your new GitHub repository (looks like `https://github.com/YOUR_USERNAME/REPO_NAME.git`) and run:

```bash
# Add the remote origin
git remote add origin https://github.com/YOUR_USERNAME/MRI-Brain-Tumor-Classification.git

# Rename the branch to main
git branch -M main

# Push the code
git push -u origin main
```

---

## 💡 Important Notes

### Dataset & Models
The `.gitignore` file I created will prevent Git from uploading:
- The `dataset/` folder (too many images for GitHub).
- The `saved_models/` folder (large `.h5` files).
- The `uploads/` and `logs/` folders.

This is **standard practice** for production code. Users who download your code will use the `requirements.txt` to set up the environment and run `train.py` to regenerate the model.

### Large File Storage (Optional)
If you *really* want to store your 50MB+ model files on GitHub, you should use **Git LFS** (Large File Storage), but it is usually better to host models on platforms like Hugging Face or Google Drive for distribution.
