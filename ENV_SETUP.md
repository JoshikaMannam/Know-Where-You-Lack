# Environment Setup

## Required Environment Variables

This project uses environment variables to keep sensitive data secure. Follow these steps:

### 1. Create Local Environment File

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 2. Fill in Your Actual Values

Edit `.env` and replace the placeholder values:

```env
# Database Configuration
DB_PASSWORD=your_actual_mysql_password

# GROQ API Key (get from https://console.groq.com/keys)
GROQ_API_KEY=your_actual_groq_api_key
```

### 3. Important Notes

- ⚠️ **NEVER commit `.env` to git** (it's already in `.gitignore`)
- ✅ `.env.example` can be committed (contains no secrets)
- ✅ `application.properties` uses environment variables as defaults

### 4. How It Works

The `application.properties` file uses this pattern:
```properties
groq.api.key=${GROQ_API_KEY:default_value}
```

This means:
- If `GROQ_API_KEY` environment variable exists → use it
- Otherwise → use `default_value` (placeholder)

### 5. For Production Deployment

Set these environment variables in your hosting platform:
- Render: Dashboard → Environment → Add Environment Variable
- Heroku: Settings → Config Vars
- AWS: Environment variables in service configuration
