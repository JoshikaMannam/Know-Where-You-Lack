# 🚀 RENDER DEPLOYMENT GUIDE - KnowWhereYouLack

## 📦 **DEPLOYMENT ORDER**

1. **Database** (MySQL on Aiven/Railway/PlanetScale)
2. **Backend** (Spring Boot on Render)
3. **Frontend** (React on Render)

---

## 1️⃣ **DEPLOY DATABASE (MySQL)**

### **Option A: Aiven (FREE TIER - RECOMMENDED)**

1. Go to https://aiven.io/
2. Sign up for a free account
3. Click "Create Service" → Select "MySQL"
4. Choose:
   - **Cloud**: AWS
   - **Region**: Choose closest to you (e.g., US East)
   - **Plan**: Hobbyist (FREE)
5. Click "Create Service"
6. Wait 5-10 minutes for database to start
7. Once ready, go to "Overview" tab:
   - **Host**: Copy the host (e.g., `mysql-xxxxx.aivencloud.com`)
   - **Port**: Usually `12691`
   - **Username**: `avnadmin`
   - **Password**: Copy the password
   - **Database**: `defaultdb`

8. **Create your database**:
   - Go to "Databases" tab
   - Click "Add Database"
   - Name: `knowwhereyoulack`
   - Click "Create"

9. **Your DB_URL will be**:
   ```
   jdbc:mysql://[HOST]:[PORT]/knowwhereyoulack?sslMode=REQUIRED
   ```
   Example:
   ```
   jdbc:mysql://mysql-12345-johndoe.aivencloud.com:12691/knowwhereyoulack?sslMode=REQUIRED
   ```

### **Option B: Railway (FREE TIER)**

1. Go to https://railway.app/
2. Sign up with GitHub
3. Click "New Project" → "Provision MySQL"
4. Once created, click on MySQL service
5. Go to "Variables" tab
6. Copy these values:
   - MYSQL_HOST
   - MYSQL_PORT
   - MYSQL_USER
   - MYSQL_PASSWORD
   - MYSQL_DATABASE

7. Construct your DB_URL:
   ```
   jdbc:mysql://[MYSQL_HOST]:[MYSQL_PORT]/[MYSQL_DATABASE]
   ```

### **Option C: PlanetScale (FREE TIER)**

1. Go to https://planetscale.com/
2. Sign up
3. Create new database: `knowwhereyoulack`
4. Go to "Connect" → Choose "Java"
5. Copy the JDBC URL provided

---

## 2️⃣ **DEPLOY BACKEND (Spring Boot on Render)**

### **Step 1: Prepare GitHub Repository**

1. Make sure all your code is pushed to GitHub:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin feature/authentication-system
   ```

### **Step 2: Create Render Account**

1. Go to https://render.com/
2. Sign up with GitHub account
3. Authorize Render to access your repositories

### **Step 3: Create Web Service**

1. Click "New +" → "Web Service"
2. Connect to your GitHub repository: `Know-Where-You-Lack`
3. Configure the service:

   **Basic Settings:**
   - **Name**: `knowwhereyoulack-backend`
   - **Region**: Choose closest to you
   - **Branch**: `feature/authentication-system`
   - **Root Directory**: `java-backend`
   - **Runtime**: `Java`
   - **Build Command**: `mvn clean install -DskipTests`
   - **Start Command**: `java -jar target/backend-1.0.0.jar`

   **Instance Type:**
   - Select **Free** (0.1 CPU, 512 MB RAM)

### **Step 4: Add Environment Variables**

Click "Advanced" → "Add Environment Variable" and add these:

1. **DB_URL**
   ```
   jdbc:mysql://[YOUR_DB_HOST]:[PORT]/knowwhereyoulack?sslMode=REQUIRED
   ```
   *(Use the URL from Step 1)*

2. **DB_USERNAME**
   ```
   avnadmin
   ```
   *(Or your database username)*

3. **DB_PASSWORD**
   ```
   [Your database password from Step 1]
   ```

4. **JWT_SECRET**
   ```
   YourVerySecureSecretKeyThatIsAtLeast32CharactersLong123456
   ```
   *(Or generate a new one)*

5. **GROQ_API_KEY**
   ```
   YOUR_GROQ_API_KEY_HERE
   ```

6. **PORT**
   ```
   8082
   ```

### **Step 5: Deploy**

1. Click "Create Web Service"
2. Wait 10-15 minutes for first deployment
3. Watch the logs for any errors
4. Once deployed, you'll get a URL like: `https://knowwhereyoulack-backend.onrender.com`

### **Step 6: Test Backend**

Open in browser:
```
https://knowwhereyoulack-backend.onrender.com/api/auth/test
```

Or test login endpoint with Postman/Thunder Client.

---

## 3️⃣ **DEPLOY FRONTEND (React on Render)**

### **Step 1: Update Frontend API URL**

Before deploying frontend, you need to update the API base URL to point to your deployed backend.

**Edit `frontend/src/components/Login.tsx`**:
```typescript
const API_BASE_URL = 'https://knowwhereyoulack-backend.onrender.com/api';
```

**Edit any other files that use `http://localhost:8082`** and change to your backend URL.

### **Step 2: Create Frontend Web Service**

1. In Render dashboard, click "New +" → "Static Site"
2. Connect to same GitHub repository
3. Configure:

   **Basic Settings:**
   - **Name**: `knowwhereyoulack-frontend`
   - **Branch**: `feature/authentication-system`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`

### **Step 3: Add Environment Variable (if needed)**

If you have any environment variables in frontend:
- **VITE_API_URL**: `https://knowwhereyoulack-backend.onrender.com`

### **Step 4: Deploy**

1. Click "Create Static Site"
2. Wait 5-10 minutes
3. You'll get a URL like: `https://knowwhereyoulack-frontend.onrender.com`

---

## ✅ **POST-DEPLOYMENT CHECKLIST**

- [ ] Database is running and accessible
- [ ] Backend deployed and responding to API calls
- [ ] Frontend deployed and can load
- [ ] Login/Signup works end-to-end
- [ ] CORS is properly configured (backend should allow frontend URL)

---

## 🔧 **TROUBLESHOOTING**

### **Backend won't start:**
1. Check logs in Render dashboard
2. Verify all environment variables are set
3. Ensure DB_URL is correct and database is accessible
4. Check if Java version matches (should be Java 21)

### **Database connection failed:**
1. Verify database is running
2. Check DB_URL format
3. Ensure database allows connections from external IPs
4. Check username/password

### **Frontend can't connect to backend:**
1. Update API_BASE_URL in frontend code
2. Add CORS configuration in backend:
   - Allow your frontend domain
   - Check `WebConfig.java` or similar

### **CORS errors:**
Backend needs to allow requests from your frontend domain. 

---

## 📝 **IMPORTANT NOTES**

1. **Free Tier Limitations:**
   - Backend goes to sleep after 15 minutes of inactivity
   - First request after sleep takes 50+ seconds to wake up
   - Consider keeping it alive with a cron job or upgrade to paid plan

2. **Database:**
   - Free tier databases have connection limits
   - May need to adjust connection pool settings in Spring Boot

3. **Environment Variables:**
   - Never commit secrets to GitHub
   - Always use environment variables for sensitive data

4. **Custom Domain (Optional):**
   - You can add a custom domain in Render settings
   - Both frontend and backend support custom domains

---

## 🎉 **NEXT STEPS AFTER READING THIS**

1. **Tell me which database option you want to use** (Aiven, Railway, or PlanetScale)
2. I'll help you configure it
3. Then we'll deploy backend
4. Finally, deploy frontend

**Ready to start? Which database service do you prefer?**
