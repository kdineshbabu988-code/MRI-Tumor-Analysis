# Deployment Guide for Render

This project is configured for deployment on Render.com.

## Prerequisites

1. Push your code to a GitHub/GitLab repository.
2. Sign up for a [Render](https://render.com) account.

## Option 1: Blueprints (Recommended)

1. In Render Dashboard, click **New +** -> **Blueprint**.
2. Connect your repository.
3. Render will automatically detect the `render.yaml` file and configure the service for you.
4. Click **Apply**.

## Option 2: Manual Setup (Web Service)

1. Click **New +** -> **Web Service**.
2. Connect your repository.
3. Use the following settings:
   - **Name**: mri-tumor-classification
   - **Language**: Python 3
   - **Branch**: main (or master)
   - **Root Directory**: . (Leave empty or set to `.`)
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && gunicorn wsgi:app`
4. Click **Create Web Service**.

## Notes

- This app uses `gunicorn` for production serving.
- Files uploaded by users are ephemeral (will be lost on restart) unless you use a persistent disk (Render Disk), but for this demo, standard ephemeral storage is fine.
