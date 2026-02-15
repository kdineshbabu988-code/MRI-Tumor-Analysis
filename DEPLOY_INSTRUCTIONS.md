
# 🚀 deployment Guide: Brain Tumor Classification on Firebase

I have prepared your project for Firebase Hosting (Frontend) and Cloud Functions (Backend). 
Since you don't have Node.js installed, you need to follow these steps to deploy.

## ✅ Step 1: Install Node.js
Firebase tools require Node.js.
1. Download and install **Node.js (LTS version)** from: https://nodejs.org/
2. After installation, restart your terminal/cmd.

## ✅ Step 2: Install Firebase CLI
Open a new terminal (CMD or PowerShell) and run:
```sh
npm install -g firebase-tools
```

## ✅ Step 3: Login to Firebase
Authenticate with your Google account:
```sh
firebase login
```

## ✅ Step 4: Initialize Project
Link this folder to your Firebase project:
```sh
firebase init
```
1. Select **Hosting** and **Functions** (use Spacebar to select, Enter to confirm).
2. Select **Use an existing project**.
3. Choose your project from the list.
4. **Functions Setup**:
   - Language: **Python**
   - Do you want to use the virtualenv? **No** (Cloud handles it)
   - Do you want to install dependencies now? **No**
   - **IMPORTANT**: If it asks to overwrite `functions/main.py`, `functions/requirements.txt`, or `firebase.json`, select **NO**.
5. **Hosting Setup**:
   - Public directory: **public** (Type `public` and press Enter)
   - Configure as single-page app? **Yes**
   - Set up automatic builds? **No**
   - **IMPORTANT**: If it asks to overwrite `public/index.html`, select **NO**.

## ✅ Step 5: Deploy
Run the deployment command:
```sh
firebase deploy
```

## 🎉 Success!
After deployment, Firebase will give you a **Hosting URL** (e.g., `https://your-project.web.app`).
- Open that link to use your app.
- The first prediction might take ~10-15 seconds (Cold Start), but subsequent ones will be fast.

---

## 📂 Project Structure Created due to your request
- `public/`: Contains your HTML, CSS, JS (Frontend).
- `functions/`: Contains your Python backend and Model.
- `firebase.json`: Configuration for routing `/predict` to the backend.
