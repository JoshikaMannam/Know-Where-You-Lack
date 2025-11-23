# 🚀 RENDER DEPLOYMENT GUIDE - ALL IN RENDER

**Deploy your complete "Know Where You Lack" application using ONLY Render services**

---

## 📋 **WHAT YOU'LL DEPLOY**

1. **MySQL Database** (Render) - FREE
2. **Backend** (Spring Boot) - FREE
3. **Frontend** (React) - FREE

**Total Cost: $0** | **Total Time: ~25 minutes**

---

## ✅ **STEP 1: Deploy MySQL Database on Render (5 minutes)**

### Create Database

1. **Go to Render Dashboard**
   - URL: https://render.com/
   - Sign up with GitHub (if not already)

2. **Create PostgreSQL Database** (Render's free database)
   - Click "New +" → "PostgreSQL"
   - Name: `knowwhereyoulack-db`
   - Database: `knowwhereyoulack`
   - User: `knowwhereyoulack_user` (auto-generated)
   - Region: **Oregon (US West)** or closest to you
   - Plan: **Free**
   - Click "Create Database"

3. **Wait 2-3 minutes** for provisioning

4. **Get Connection Details**
   - Click on your database service
   - Scroll to "Connections"
   - You'll see:
     - **Internal Database URL** (for services in same region)
     - **External Database URL** (for external connections)

5. **Copy Internal Database URL**
   ```
   postgresql://knowwhereyoulack_user:xxxxx@dpg-xxxxx.oregon-postgres.render.com/knowwhereyoulack
   ```

### ⚠️ **IMPORTANT: Update Backend for PostgreSQL**

Since Render uses PostgreSQL (not MySQL), we need to update your backend:

**You need to:**
1. Update `pom.xml` - Replace MySQL driver with PostgreSQL
2. Update `application.properties` - Change dialect and driver

---

## 🔧 **STEP 2: Update Backend for PostgreSQL (2 minutes)**

### Update pom.xml

Replace MySQL dependency with PostgreSQL:

**Find this in `java-backend/pom.xml`:**
```xml
<!-- MySQL Driver -->
<dependency>
    <groupId>com.mysql</groupId>
    <artifactId>mysql-connector-j</artifactId>
    <scope>runtime</scope>
</dependency>
```

**Replace with:**
```xml
<!-- PostgreSQL Driver -->
<dependency>
    <groupId>org.postgresql</groupId>
    <artifactId>postgresql</artifactId>
    <scope>runtime</scope>
</dependency>
```

### Update application.properties

**Change these lines in `java-backend/src/main/resources/application.properties`:**

**FROM (MySQL):**
```properties
spring.datasource.url=${DB_URL:jdbc:mysql://localhost:3306/knowwhereyoulack}
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.MySQLDialect
```

**TO (PostgreSQL):**
```properties
spring.datasource.url=${DATABASE_URL:jdbc:postgresql://localhost:5432/knowwhereyoulack}
spring.datasource.driver-class-name=org.postgresql.Driver
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.PostgreSQLDialect
```

### Commit Changes

```powershell
cd D:\Know-Where-You-Lack
git add .
git commit -m "Update backend to use PostgreSQL for Render deployment"
git push origin main
```

---

## ✅ **STEP 3: Deploy Backend to Render (10 minutes)**

### Create Web Service

1. **In Render Dashboard**
   - Click "New +" → "Web Service"
   - Connect GitHub repository: `Know-Where-You-Lack`
   - Click "Connect"

2. **Configure Service**
   ```
   Name: knowwhereyoulack-backend
   Region: Oregon (US West) - SAME as database
   Branch: main
   Root Directory: java-backend
   Runtime: Java
   Build Command: mvn clean install -DskipTests
   Start Command: java -jar target/backend-1.0.0.jar
   Instance Type: Free
   ```

3. **Add Environment Variables** (Click "Advanced")

   | Key | Value | Notes |
   |-----|-------|-------|
   | `DATABASE_URL` | `[Your PostgreSQL Internal URL from Step 1]` | From database connection info |
   | `JWT_SECRET` | `YourVerySecureSecretKeyThatIsAtLeast32CharactersLong123456` | Or generate new one |
   | `GROQ_API_KEY` | `[Your GROQ API Key from .env file]` | Your GROQ key |
   | `PORT` | `8082` | Required |

   **To get your DATABASE_URL:**
   - Go to your database service
   - Click "Info" tab
   - Copy "Internal Database URL"
   - Paste into `DATABASE_URL` variable

   **Note**: Render automatically parses `DATABASE_URL` into username, password, host, etc.

4. **Deploy**
   - Click "Create Web Service"
   - Wait 10-15 minutes (first build)
   - Watch logs for any errors

5. **Get Backend URL**
   - After deployment: `https://knowwhereyoulack-backend.onrender.com`
   - Test: `https://knowwhereyoulack-backend.onrender.com/api/quiz/topics`

---

## ✅ **STEP 4: Update CORS for Production (2 minutes)**

Your backend needs to allow requests from your production frontend.

**Update `java-backend/src/main/java/com/knowwhereyoulack/config/SecurityConfig.java`:**

Find this section:
```java
configuration.setAllowedOrigins(Arrays.asList(
    "http://localhost:5173", 
    "http://localhost:5174",
    "https://knowwhereyoulack-frontend.onrender.com",
    "https://knowwhereyoulack.onrender.com"
));
```

If you use a different name for your frontend, update it here.

Then commit and push:
```powershell
git add .
git commit -m "Update CORS for production frontend"
git push origin main
```

Render will auto-redeploy backend (takes ~5 min).

---

## ✅ **STEP 5: Deploy Frontend to Render (5 minutes)**

### Create Static Site

1. **In Render Dashboard**
   - Click "New +" → "Static Site"
   - Repository: `Know-Where-You-Lack` (already connected)
   - Click "Connect"

2. **Configure Static Site**
   ```
   Name: knowwhereyoulack-frontend
   Branch: main
   Root Directory: frontend
   Build Command: npm install && npm run build
   Publish Directory: dist
   ```

3. **Add Environment Variable**
   
   | Key | Value |
   |-----|-------|
   | `VITE_API_URL` | `https://knowwhereyoulack-backend.onrender.com/api` |
   
   **⚠️ IMPORTANT**: Replace with your actual backend URL from Step 3!

4. **Deploy**
   - Click "Create Static Site"
   - Wait 5-10 minutes
   - Watch build logs

5. **Get Frontend URL**
   ```
   https://knowwhereyoulack-frontend.onrender.com
   ```

---

## ✅ **STEP 6: Verification (5 minutes)**

### Test Everything

1. **Open Your Frontend**
   ```
   https://knowwhereyoulack-frontend.onrender.com
   ```

2. **Test Authentication** ✅
   - Click "Login/Signup" tab
   - Register new account
   - Login with credentials
   - Should redirect to dashboard

3. **Test Quiz System** ✅
   - Click "Quizzes"
   - Select a topic
   - Start quiz
   - Answer questions
   - See results

4. **Test Skilli Chatbot** ✅
   - Click "Chatbot"
   - Ask: "What is Object-Oriented Programming?"
   - Get AI response

5. **Test Notes** ✅
   - Go to Notes section
   - Create a note
   - Edit note
   - Delete note

6. **Check Dashboard** ✅
   - View analytics
   - See weak topics
   - View quiz accuracy chart

---

## 🐛 **TROUBLESHOOTING**

### Backend Won't Start

1. **Check Build Logs**
   - Go to backend service → Logs tab
   - Look for errors

2. **Common Issues:**

   **PostgreSQL Connection Failed**
   ```
   Error: Connection refused
   ```
   **Fix:**
   - Verify `DATABASE_URL` is set correctly
   - Check database service is running
   - Ensure backend and database are in same region

   **Missing PostgreSQL Driver**
   ```
   Error: No suitable driver found
   ```
   **Fix:**
   - Verify you updated `pom.xml` with PostgreSQL dependency
   - Commit and push changes
   - Redeploy

   **GROQ API Key Missing**
   ```
   Error: GROQ_API_KEY not found
   ```
   **Fix:**
   - Go to backend service → Environment tab
   - Add `GROQ_API_KEY` variable
   - Redeploy

### Frontend Can't Connect

1. **CORS Error in Browser Console**
   ```
   Access to XMLHttpRequest blocked by CORS
   ```
   **Fix:**
   - Verify `VITE_API_URL` in frontend environment variables
   - Check `SecurityConfig.java` has your frontend URL
   - Redeploy backend

2. **404 Not Found**
   ```
   GET https://...backend.../api/quiz/topics 404
   ```
   **Fix:**
   - Check backend is running (green status)
   - Wait if backend is waking up (~50 sec cold start)
   - Verify backend URL in `VITE_API_URL`

### Database Issues

1. **Tables Not Created**
   - Check backend logs for JPA/Hibernate errors
   - Verify `spring.jpa.hibernate.ddl-auto=update` in `application.properties`
   - Restart backend service

2. **Connection Pool Exhausted**
   - Free tier has connection limits
   - Check for connection leaks in code
   - Consider upgrading plan

---

## 📊 **RENDER FREE TIER LIMITS**

### What You Get FREE:

1. **PostgreSQL Database**
   - 1 GB storage
   - Expires after 90 days (backup and recreate)
   - Good for development/demo

2. **Web Service (Backend)**
   - 750 hours/month (enough for 1 service 24/7)
   - 512 MB RAM
   - Sleeps after 15 min inactivity
   - First request after sleep: 50-60 sec

3. **Static Site (Frontend)**
   - 100 GB bandwidth/month
   - No sleep/downtime
   - Always fast

### Keeping Backend Awake (Optional)

Use a free service to ping your backend:

1. **UptimeRobot** (https://uptimerobot.com/)
   - Free plan: 50 monitors
   - Ping every 5 minutes
   - URL to monitor: `https://your-backend.onrender.com/api/quiz/topics`

2. **Cron-Job.org** (https://cron-job.org/)
   - Free scheduled requests
   - Ping every 5-10 minutes

---

## 🔄 **UPDATING YOUR DEPLOYED APP**

Whenever you make changes:

```powershell
# 1. Make changes locally
# 2. Test locally
# 3. Commit and push
git add .
git commit -m "Your update description"
git push origin main

# 4. Render auto-deploys (5-15 minutes)
```

**Auto-Deploy is enabled by default:**
- Backend: Rebuilds on every push to main
- Frontend: Rebuilds on every push to main
- Database: Manual updates only

---

## ✅ **SUCCESS CHECKLIST**

- [ ] PostgreSQL database created on Render
- [ ] Backend updated for PostgreSQL (pom.xml + application.properties)
- [ ] Backend deployed on Render
- [ ] Backend responding at `/api/quiz/topics`
- [ ] Frontend deployed on Render
- [ ] Can access frontend URL
- [ ] Login/Registration working
- [ ] Quiz system functional
- [ ] Skilli chatbot responding
- [ ] Notes CRUD working
- [ ] Dashboard showing data

---

## 🎊 **YOU'RE DONE!**

Your app is now live at:
- **Frontend**: https://knowwhereyoulack-frontend.onrender.com
- **Backend**: https://knowwhereyoulack-backend.onrender.com
- **Database**: PostgreSQL on Render

**Total Cost: $0** | **All on Render platform!**

Share your frontend URL with anyone! 🚀

---

## 📝 **IMPORTANT NOTES**

### Environment Variables Summary

**Backend (Web Service):**
```
DATABASE_URL=postgresql://user:pass@host/dbname
JWT_SECRET=YourVerySecureSecretKeyThatIsAtLeast32CharactersLong123456
GROQ_API_KEY=[Your GROQ API Key]
PORT=8082
```

**Frontend (Static Site):**
```
VITE_API_URL=https://knowwhereyoulack-backend.onrender.com/api
```

### Database Connection

Render provides `DATABASE_URL` in this format:
```
postgresql://username:password@host:port/database
```

Spring Boot automatically parses this when you use `${DATABASE_URL}` in `application.properties`.

### First Deploy Tips

- **Be Patient**: First deploy takes 10-15 minutes
- **Watch Logs**: Very detailed, helps debug issues
- **Test Locally First**: Make sure everything works before deploying
- **Check Health**: Backend `/api/quiz/topics` should return JSON

---

## 🆘 **NEED HELP?**

1. Check Render logs (most issues show here)
2. Verify environment variables are set
3. Test backend endpoint directly
4. Check browser console for frontend errors
5. Ensure database and backend are in same region

**Common mistake**: Forgetting to update pom.xml and application.properties for PostgreSQL!

---

## 🎯 **NEXT STEPS**

After successful deployment:

1. ✅ Test all features thoroughly
2. ✅ Share your app URL
3. ✅ Monitor Render dashboard
4. ✅ Set up UptimeRobot (optional)
5. ✅ Consider custom domain (optional)

**Congratulations on deploying your app! 🎉**
