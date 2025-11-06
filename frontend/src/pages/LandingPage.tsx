import React, { useState, useEffect } from 'react';
import '../styles/Landing.css';
import Login from '../components/Login';
import { 
  Brain, 
  Target, 
  Bot, 
  BarChart, 
  BookOpen, 
  Clock,
  Mail,
  MapPin,
  Github,
  Linkedin,
  CheckCircle,
  Sparkles,
  Zap,
  Users,
  Sun,
  Moon
} from 'lucide-react';

type Tab = 'home' | 'about' | 'contact' | 'login';
type Theme = 'light' | 'dark';

interface LandingPageProps {
  onLoginSuccess: (user: any) => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onLoginSuccess }) => {
  const [activeTab, setActiveTab] = useState<Tab>('home');
  const [theme, setTheme] = useState<Theme>('light');

  // Initialize theme from localStorage or system preference
  useEffect(() => {
    const savedTheme = localStorage.getItem('landing-theme') as Theme;
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const initialTheme = savedTheme || systemTheme;
    setTheme(initialTheme);
    document.documentElement.setAttribute('data-theme', initialTheme);
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('landing-theme', newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <HomeTab onGetStarted={() => setActiveTab('login')} />;
      case 'about':
        return <AboutTab />;
      case 'contact':
        return <ContactTab />;
      case 'login':
        return <LoginTab onLoginSuccess={onLoginSuccess} />;
      default:
        return <HomeTab onGetStarted={() => setActiveTab('login')} />;
    }
  };

  return (
    <div className="landing-container">
      {/* Navigation Bar */}
      <nav className="landing-navbar">
        <div className="navbar-content">
          {/* Left: Project Names */}
          <div className="navbar-brand">
            <div className="brand-primary">
              <span className="brand-icon">🎓</span>
              <span className="brand-name">KnowWhereYouLack</span>
            </div>
            <div className="brand-secondary">LearnMate</div>
          </div>

          {/* Right: Navigation Tabs + Theme Toggle */}
          <div className="navbar-right">
            <div className="navbar-tabs">
              <button 
                className={`nav-tab ${activeTab === 'home' ? 'active' : ''}`}
                onClick={() => setActiveTab('home')}
              >
                Home
              </button>
              <button 
                className={`nav-tab ${activeTab === 'about' ? 'active' : ''}`}
                onClick={() => setActiveTab('about')}
              >
                About
              </button>
              <button 
                className={`nav-tab ${activeTab === 'contact' ? 'active' : ''}`}
                onClick={() => setActiveTab('contact')}
              >
                Contact
              </button>
              <button 
                className={`nav-tab nav-tab-cta ${activeTab === 'login' ? 'active' : ''}`}
                onClick={() => setActiveTab('login')}
              >
                Login / Signup
              </button>
            </div>
            
            {/* Theme Toggle Button */}
            <button 
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label="Toggle theme"
            >
              {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
            </button>
          </div>
        </div>
      </nav>

      {/* Content Area */}
      <main className="landing-content">
        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="landing-footer">
        <p>&copy; 2025 KnowWhereYouLack | Built with ❤️ by Team LearnMate</p>
        <p className="footer-tech">AI/ML • OOPS • Full Stack</p>
      </footer>
    </div>
  );
};

// ============================================
// HOME TAB COMPONENT
// ============================================
const HomeTab: React.FC<{ onGetStarted: () => void }> = ({ onGetStarted }) => {
  return (
    <div className="home-tab">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-badge">
            <Sparkles className="badge-icon" />
            <span>AI-Powered Learning Platform</span>
          </div>
          
          <h1 className="hero-title">
            Discover Your <span className="gradient-text">Learning Gaps</span>
            <br />
            Master Every Topic
          </h1>
          
          <p className="hero-subtitle">
            An intelligent, context-aware platform that learns from the learner, not just the data.
            Transform passive learning into active, insight-driven growth.
          </p>
          
          <div className="hero-buttons">
            <button className="btn-primary" onClick={onGetStarted}>
              <span>Get Started Free</span>
              <Zap className="btn-icon" />
            </button>
          </div>

          {/* Stats */}
          <div className="hero-stats">
            <div className="stat-card">
              <div className="stat-number">270+</div>
              <div className="stat-label">Quiz Questions</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">9</div>
              <div className="stat-label">Subjects</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">87.8%</div>
              <div className="stat-label">ML Accuracy</div>
            </div>
          </div>
        </div>

        {/* Floating Cards */}
        <div className="hero-visual">
          <div className="float-card card-1">
            <BarChart className="card-icon" />
            <h3>Performance Analytics</h3>
            <p>Track your progress with detailed insights</p>
          </div>
          <div className="float-card card-2">
            <Bot className="card-icon" />
            <h3>AI Tutor - Skilli</h3>
            <p>Get instant help from our AI assistant</p>
          </div>
          <div className="float-card card-3">
            <Target className="card-icon" />
            <h3>Adaptive Quizzes</h3>
            <p>Personalized difficulty levels</p>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <h2 className="section-title">Everything You Need to Excel</h2>
        <p className="section-subtitle">Powered by AI/ML and built with modern full-stack technologies</p>
        
        <div className="features-grid">
          <FeatureCard 
            icon={<Target />}
            title="Weakness Identification"
            description="ML algorithms analyze your quiz performance to pinpoint exact topics where you struggle."
            color="#FF6B6B"
          />
          <FeatureCard 
            icon={<Brain />}
            title="Adaptive Quizzes"
            description="270+ questions across 9 subjects with difficulty levels that adapt to your performance."
            color="#4A90E2"
          />
          <FeatureCard 
            icon={<Bot />}
            title="AI Tutor - Skilli"
            description="Chat with our intelligent AI assistant for instant explanations and learning support."
            color="#6C63FF"
          />
          <FeatureCard 
            icon={<BarChart />}
            title="Visual Analytics"
            description="Beautiful charts and graphs showing your progress, weak topics, and improvement trends."
            color="#2ECC71"
          />
          <FeatureCard 
            icon={<BookOpen />}
            title="Curated Resources"
            description="Get personalized learning materials: videos, articles, and notes for weak topics."
            color="#FFD93D"
          />
          <FeatureCard 
            icon={<Clock />}
            title="Study Timer"
            description="Built-in Pomodoro timer to help you maintain focus and track study sessions."
            color="#FF6B6B"
          />
        </div>
      </section>
    </div>
  );
};

const FeatureCard: React.FC<{ 
  icon: React.ReactNode; 
  title: string; 
  description: string;
  color: string;
}> = ({ icon, title, description, color }) => {
  return (
    <div className="feature-card" style={{ '--feature-color': color } as React.CSSProperties}>
      <div className="feature-icon">{icon}</div>
      <h3 className="feature-title">{title}</h3>
      <p className="feature-description">{description}</p>
      <div className="feature-glow"></div>
    </div>
  );
};

// ============================================
// ABOUT TAB COMPONENT
// ============================================
const AboutTab: React.FC = () => {
  const team = [
    { name: 'Joshika Mannam', roll: '2410030108', color: 'var(--team-color-1)' },
    { name: 'B. Komala Manaswini', roll: '2410030104', color: 'var(--team-color-2)' },
    { name: 'Kulkarni Sahithi', roll: '2410030057', color: 'var(--team-color-3)' },
    { name: 'Aarushi Chakraborty', roll: '2410030008', color: 'var(--team-color-4)' },
    { name: 'P. Lalitha Preethi', roll: '2410030103', color: 'var(--team-color-5)' },
  ];

  return (
    <div className="about-tab">
      <section className="about-hero">
        <h1 className="page-title">Redefining Personalized Learning</h1>
        <p className="page-subtitle">
          An intelligent learning platform combining AI/ML and Object-Oriented Programming
        </p>
      </section>

      {/* Problem & Solution */}
      <section className="problem-solution">
        <div className="content-box problem-box">
          <h2 className="box-title">🎯 The Problem We Solve</h2>
          <ul className="problem-list">
            <li><CheckCircle className="list-icon error" /> Most education systems follow a "one-size-fits-all" approach</li>
            <li><CheckCircle className="list-icon error" /> Students struggle to identify what exactly they don't understand</li>
            <li><CheckCircle className="list-icon error" /> Existing tools don't track progress or suggest improvement paths</li>
          </ul>
        </div>

        <div className="content-box solution-box">
          <h2 className="box-title">✅ Our Solution</h2>
          <ul className="solution-list">
            <li><CheckCircle className="list-icon success" /> ML models predict weak topics with 87.8% accuracy</li>
            <li><CheckCircle className="list-icon success" /> Adaptive quizzes personalized to each student's level</li>
            <li><CheckCircle className="list-icon success" /> AI-powered chatbot provides instant educational support</li>
            <li><CheckCircle className="list-icon success" /> Visual analytics track progress and build confidence</li>
          </ul>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="tech-stack">
        <h2 className="section-title">Technology Stack</h2>
        <div className="tech-grid">
          <div className="tech-card">
            <h3>🎨 Frontend</h3>
            <div className="tech-tags">
              <span className="tech-tag">React 18</span>
              <span className="tech-tag">TypeScript</span>
              <span className="tech-tag">Vite</span>
              <span className="tech-tag">Tailwind CSS</span>
            </div>
          </div>
          <div className="tech-card">
            <h3>⚙️ Backend (OOPS)</h3>
            <div className="tech-tags">
              <span className="tech-tag">Spring Boot</span>
              <span className="tech-tag">Java</span>
              <span className="tech-tag">JWT Auth</span>
              <span className="tech-tag">MySQL</span>
            </div>
          </div>
          <div className="tech-card">
            <h3>🤖 AI/ML</h3>
            <div className="tech-tags">
              <span className="tech-tag">Python</span>
              <span className="tech-tag">Scikit-learn</span>
              <span className="tech-tag">Random Forest</span>
              <span className="tech-tag">XGBoost</span>
              <span className="tech-tag">Groq API</span>
            </div>
          </div>
        </div>

        <div className="accuracy-stats">
          <div className="stat-box">
            <div className="stat-big">49.37%</div>
            <p>Early Prediction Accuracy<br/>(without final grades)</p>
          </div>
          <div className="stat-box">
            <div className="stat-big">87.80%</div>
            <p>Behavioral Analysis<br/>Prediction Accuracy</p>
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="team-section">
        <h2 className="section-title">Meet the Developers</h2>
        <p className="section-subtitle">B.Tech CSE students at KL University</p>
        
        <div className="team-grid">
          {team.map((member, index) => (
            <div key={index} className="team-card" style={{ '--team-color': member.color } as React.CSSProperties}>
              <div className="team-avatar">
                <Users className="avatar-icon" />
              </div>
              <h3 className="team-name">{member.name}</h3>
              <p className="team-roll">{member.roll}</p>
              <div className="team-socials">
                <button className="social-btn"><Linkedin size={18} /></button>
                <button className="social-btn"><Github size={18} /></button>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

// ============================================
// CONTACT TAB COMPONENT
// ============================================
const ContactTab: React.FC = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Form submitted:', formData);
    setSubmitted(true);
    setTimeout(() => {
      setSubmitted(false);
      setFormData({ name: '', email: '', subject: '', message: '' });
    }, 3000);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="contact-tab">
      <section className="contact-hero">
        <h1 className="page-title">Get in Touch</h1>
        <p className="page-subtitle">Have questions? We'd love to hear from you!</p>
      </section>

      <div className="contact-container">
        {/* Contact Info */}
        <div className="contact-info">
          <div className="info-card">
            <Mail className="info-icon" />
            <h3>Email</h3>
            <p>knowwhereyoulack@klu.ac.in</p>
          </div>
          <div className="info-card">
            <MapPin className="info-icon" />
            <h3>Institution</h3>
            <p>KL University<br/>Vijayawada, India</p>
          </div>
          <div className="info-card">
            <Brain className="info-icon" />
            <h3>Project Type</h3>
            <p>Academic Research<br/>AI/ML + Full Stack</p>
          </div>
        </div>

        {/* Contact Form */}
        <form className="contact-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Your Name</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="John Doe"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="email">Your Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="john@example.com"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="subject">Subject</label>
            <input
              type="text"
              id="subject"
              name="subject"
              value={formData.subject}
              onChange={handleChange}
              placeholder="How can we help?"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="message">Message</label>
            <textarea
              id="message"
              name="message"
              value={formData.message}
              onChange={handleChange}
              placeholder="Your message here..."
              rows={6}
              required
            />
          </div>

          <button type="submit" className="btn-submit">
            {submitted ? '✅ Message Sent!' : 'Send Message'}
          </button>
        </form>
      </div>
    </div>
  );
};

// ============================================
// LOGIN TAB COMPONENT
// ============================================
const LoginTab: React.FC<{ onLoginSuccess: (user: any) => void }> = ({ onLoginSuccess }) => {
  return (
    <div className="login-tab">
      <Login onLoginSuccess={onLoginSuccess} />
    </div>
  );
};

export default LandingPage;