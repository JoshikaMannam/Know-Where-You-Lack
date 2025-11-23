# 🚀 Complete Deployment Guide - Know Where You Lack

## 📋 Pre-Deployment Checklist

- [x] Authentication system working locally
- [x] Environment variables configured (.env files)
- [x] API endpoints tested
- [x] CORS configured for production
- [x] Git repository pushed to GitHub
- [ ] Database provisioned
- [ ] Backend deployed
- [ ] Frontend deployed

---

## 🎯 Deployment Strategy

### Architecture Overview
```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   Frontend  │─────▶│   Backend    │─────▶│   Database   │
│  (Render)   │      │  (Render)    │      │   (Aiven)    │
│   Static    │      │  Spring Boot │      │    MySQL     │
└─────────────┘      └──────────────┘      └──────────────┘
```

---

## STEP 1: Provision Database (MySQL)

### Option A: Aiven (RECOMMENDED - Easiest Setup)

1. **Sign Up**
   - Go to https://aiven.io/
   - Sign up for free account

2. **Create MySQL Service**
   - Click "Create Service"
   - Select **MySQL**
   - Cloud: **AWS**
   - Region: **US East (N. Virginia)** or closest to you
   - Plan: **Hobbyist** (FREE)
   - Service Name: `knowwhereyoulack-db`
   - Click "Create Service"

3. **Wait for Provisioning** (5-10 minutes)
   - Service status will change to "RUNNING"

4. **Get Connection Details**
   - Go to "Overview" tab
   - Copy these values:
     - **Host**: `mysql-xxxxx-yourname.aivencloud.com`
     - **Port**: `12691` (or shown port)
     - **User**: `avnadmin`
     - **Password**: Click "Show" and copy
     - **Database**: `defaultdb`

5. **Create Database**
   - Go to "Databases" tab
   - Click "Add database"
   - Name: `knowwhereyoulack`
   - Click "Create"

6. **Your Connection String**
   ```
   jdbc:mysql://mysql-xxxxx-yourname.aivencloud.com:12691/knowwhereyoulack?sslMode=REQUIRED
   ```

### Option B: Railway

1. Go to https://railway.app/
2. Sign up with GitHub
3. Create "New Project" → "Provision MySQL"
4. Copy connection details from Variables tab

### Option C: Supabase (PostgreSQL Alternative)

If you want to use PostgreSQL instead:
1. Go to https://supabase.com/
2. Create new project
3. Note: You'll need to modify `application.properties` for PostgreSQL

---

## STEP 2: Deploy Backend (Spring Boot on Render)

### 2.1: Update Backend for Production

The backend is already configured! Environment variables pattern is ready:
```properties
spring.datasource.url=${DB_URL:jdbc:mysql://localhost:3306/knowwhereyoulack}
spring.datasource.password=${DB_PASSWORD:your_mysql_password}
groq.api.key=${GROQ_API_KEY:your_groq_api_key_here}
```

### 2.2: Create Render Web Service

1. **Go to Render Dashboard**
   - Visit https://render.com/
   - Sign up with GitHub
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Select `Know-Where-You-Lack` repository
   - Branch: `feature/authentication-system`

3. **Configure Service**
   ```
   Name: knowwhereyoulack-backend
   Region: Oregon (or closest to your DB)
   Branch: feature/authentication-system
   Root Directory: java-backend
   Runtime: Java
   Build Command: mvn clean install -DskipTests
   Start Command: java -jar target/backend-1.0.0.jar
   Instance Type: Free
   ```

4. **Environment Variables** (CRITICAL!)
   Click "Advanced" → Add these variables:

   | Key | Value |
   |-----|-------|
   | `DB_URL` | `jdbc:mysql://[YOUR-AIVEN-HOST]:[PORT]/knowwhereyoulack?sslMode=REQUIRED` |
   | `DB_USERNAME` | `avnadmin` |
   | `DB_PASSWORD` | `[Your Aiven password]` |
   | `JWT_SECRET` | `YourVerySecureSecretKeyThatIsAtLeast32CharactersLong123456` |
   | `GROQ_API_KEY` | `[Your GROQ API key from console.groq.com/keys]` |
   | `PORT` | `8082` |

5. **Deploy**
   - Click "Create Web Service"
   - Wait 10-15 minutes for first build
   - Watch logs for errors

6. **Get Backend URL**
   - After deployment: `https://knowwhereyoulack-backend.onrender.com`
   - Test: `https://knowwhereyoulack-backend.onrender.com/api/quiz/topics`

---

## STEP 3: Update Frontend for Production

### 3.1: Update Backend CORS

Backend needs to allow your frontend domain. Add to `SecurityConfig.java`:

```java
configuration.setAllowedOrigins(Arrays.asList(
    "http://localhost:5173", 
    "http://localhost:5174",
    "https://knowwhereyoulack-frontend.onrender.com"  // Add this
));
```

### 3.2: Verify Environment Files

Already created:
- ✅ `.env.development` (localhost)
- ✅ `.env.production` (Render URL)
- ✅ `src/config/api.ts` (centralized API config)

---

## STEP 4: Deploy Frontend (React on Render)

### 4.1: Create Static Site

1. **In Render Dashboard**
   - Click "New +" → "Static Site"
   - Connect same repository
   - Branch: `feature/authentication-system`

2. **Configure Static Site**
   ```
   Name: knowwhereyoulack-frontend
   Branch: feature/authentication-system
   Root Directory: frontend
   Build Command: npm install && npm run build
   Publish Directory: dist
   ```

3. **Environment Variables**
   
   | Key | Value |
   |-----|-------|
   | `VITE_API_URL` | `https://knowwhereyoulack-backend.onrender.com/api` |

4. **Deploy**
   - Click "Create Static Site"
   - Wait 5-10 minutes
   - Get URL: `https://knowwhereyoulack-frontend.onrender.com`

---

## STEP 5: Post-Deployment Configuration

### 5.1: Update Backend CORS (Again)

Now that you have the actual frontend URL:

1. Edit `SecurityConfig.java`:
   ```java
   configuration.setAllowedOrigins(Arrays.asList(
       "http://localhost:5173",
       "http://localhost:5174",
       "https://knowwhereyoulack-frontend.onrender.com"
   ));
   ```

2. Commit and push:
   ```bash
   git add java-backend/src/main/java/com/knowwhereyoulack/config/SecurityConfig.java
   git commit -m "Add production frontend URL to CORS"
   git push origin feature/authentication-system
   ```

3. Render will auto-redeploy backend

### 5.2: Test Everything

1. **Database**: Check Aiven dashboard - service running
2. **Backend**: Visit `https://knowwhereyoulack-backend.onrender.com/api/quiz/topics`
3. **Frontend**: Visit `https://knowwhereyoulack-frontend.onrender.com`
4. **Auth Flow**: Try login/signup
5. **Quiz**: Take a quiz
6. **Chatbot**: Test Skilli AI

---

## 🔧 Troubleshooting

### Backend Issues

**Build Failed**
- Check Java version in `pom.xml` matches Render's Java
- Verify all dependencies in `pom.xml`
- Check build logs in Render dashboard

**Database Connection Failed**
- Verify DB_URL format includes `?sslMode=REQUIRED` for Aiven
- Check database is running in Aiven dashboard
- Verify username/password are correct
- Ensure database `knowwhereyoulack` exists

**Environment Variables Not Loading**
- Check variable names match exactly (case-sensitive)
- Restart service after adding variables
- Check logs for "Using environment variable" messages

### Frontend Issues

**API Calls Failing**
- Verify VITE_API_URL is set correctly
- Check browser console for CORS errors
- Ensure backend CORS allows your frontend domain

**Build Failed**
- Check Node version (should be 18+)
- Verify `package.json` has all dependencies
- Check build logs in Render

**404 on Refresh**
- Add `_redirects` file to `public/` folder:
  ```
  /*    /index.html   200
  ```

### Performance Issues

**Backend Cold Start**
- Free tier sleeps after 15 min inactivity
- First request takes 50+ seconds to wake up
- Consider upgrading or using a keep-alive service

---

## 📊 Quick Command Reference

```bash
# Update code and redeploy
git add .
git commit -m "Update for production"
git push origin feature/authentication-system

# Check backend logs
# (Do this in Render dashboard)

# Test backend API
curl https://knowwhereyoulack-backend.onrender.com/api/quiz/topics

# Test frontend
curl https://knowwhereyoulack-frontend.onrender.com
```

---

## ✅ Deployment Checklist

### Pre-Deployment
- [x] Code pushed to GitHub
- [x] Environment variables configured
- [x] .env files in .gitignore
- [x] API URLs use environment variables

### Database
- [ ] MySQL service created on Aiven
- [ ] Database `knowwhereyoulack` created
- [ ] Connection details saved securely

### Backend
- [ ] Render web service created
- [ ] Environment variables added
- [ ] Build successful
- [ ] API endpoints responding
- [ ] Database tables created (auto via JPA)

### Frontend
- [ ] Static site created on Render
- [ ] VITE_API_URL configured
- [ ] Build successful
- [ ] Site loads correctly

### Integration
- [ ] Backend CORS allows frontend domain
- [ ] Login/signup working
- [ ] Quiz system functional
- [ ] Chatbot responding
- [ ] Analytics displaying

---

## 🎉 You're Done!

Your application is now live at:
- **Frontend**: https://knowwhereyoulack-frontend.onrender.com
- **Backend**: https://knowwhereyoulack-backend.onrender.com

Share the frontend URL with users!

---

## 🔄 Updating Your Deployed App

Whenever you make changes:

```bash
# 1. Make your changes locally
# 2. Test locally
# 3. Commit and push
git add .
git commit -m "Your update description"
git push origin feature/authentication-system

# 4. Render auto-deploys (takes 5-15 min)
```

---

## 💡 Next Steps

1. **Custom Domain**: Add your own domain in Render settings
2. **Analytics**: Add Google Analytics to track usage
3. **Monitoring**: Use Render's monitoring tools
4. **Scaling**: Upgrade to paid tier when needed
5. **CDN**: Consider using Cloudflare for better performance

---

**Need Help?** Check Render documentation or ask me!
