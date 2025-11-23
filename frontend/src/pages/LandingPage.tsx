import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Login from '../components/Login';
import {
  Home,
  Info,
  LogIn,
  Mail,
  Brain,
  Target,
  MessageSquare,
  TrendingUp,
  BookOpen,
  Zap,
  Clock,
  CheckCircle,
  ArrowRight,
  Menu,
  X,
  Send,
} from 'lucide-react';

type TabType = 'home' | 'about' | 'auth' | 'contact';

interface LandingPageProps {
  onLoginSuccess: (user: any) => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onLoginSuccess }) => {
  const [activeTab, setActiveTab] = useState<TabType>('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Contact form state
  const [contactForm, setContactForm] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
    phone: '',
  });

  const tabs = [
    { id: 'home' as TabType, label: 'Home', icon: Home },
    { id: 'about' as TabType, label: 'About Us', icon: Info },
    { id: 'auth' as TabType, label: 'Login/Signup', icon: LogIn },
    { id: 'contact' as TabType, label: 'Contact', icon: Mail },
  ];

  const features = [
    {
      icon: Target,
      title: 'Adaptive Quiz System',
      description: '3 difficulty levels (EASY, MEDIUM, HARD) with topic-wise question analysis',
      color: 'from-purple-500 to-pink-500',
    },
    {
      icon: MessageSquare,
      title: 'Intelligent Chatbot',
      description: 'Context-aware educational assistance powered by AI (GROQ/Gemini)',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: TrendingUp,
      title: 'ML-Powered Analytics',
      description: 'Real-time performance tracking & weak topic identification using machine learning',
      color: 'from-orange-500 to-red-500',
    },
  ];

  const keyFeatures = [
    'Complete Authentication System (JWT-based)',
    'Quiz System with 3 Difficulty Levels',
    'Study Notes Management',
    'Pomodoro-Style Study Timer',
    'AI Chatbot (Skilli)',
    'Real-time Analytics Dashboard',
    'Beautiful UI with Dark/Light Theme',
    'MySQL Database with 15+ Tables',
    '50+ RESTful API Endpoints',
    'ML-Powered Weakness Detection',
  ];

  const faqs = [
    {
      q: 'How does Know Where You Lack identify weak topics?',
      a: 'Using ensemble machine learning models (Random Forest + XGBoost), we analyze your quiz performance, study behavior, and engagement patterns to classify your understanding level for each topic.',
    },
    {
      q: 'Can I get help from the AI chatbot at any time?',
      a: "Yes! Our Skilli AI chatbot is available 24/7 to answer your educational questions with context-aware responses.",
    },
    {
      q: 'How accurate is the weakness prediction?',
      a: 'Our model achieves 87.80% accuracy on behavioral data, comparable to state-of-the-art research while being privacy-focused.',
    },
    {
      q: 'Which subjects are supported?',
      a: 'Currently: OOP, DSA, Physics, Chemistry, Mathematics, Operating Systems, AI/ML, and Biology. More subjects coming soon!',
    },
    {
      q: 'Is my data secure?',
      a: 'Yes! We use JWT-based authentication, encrypted passwords (BCrypt), and comply with data privacy standards.',
    },
  ];

  const handleContact = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Contact:', contactForm);
    // TODO: Add contact form submission
    alert('Thank you for contacting us! We will get back to you soon.');
    setContactForm({ name: '', email: '', subject: '', message: '', phone: '' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/80 backdrop-blur-lg border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Know Where You Lack
              </span>
            </div>

            {/* Desktop Tabs */}
            <div className="hidden md:flex space-x-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                    activeTab === tab.id
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                      : 'text-gray-300 hover:text-white hover:bg-slate-800'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-slate-800"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-slate-900 border-t border-purple-500/20">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center space-x-2 px-4 py-3 transition-all ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                    : 'text-gray-300 hover:bg-slate-800'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        )}
      </nav>

      {/* Content */}
      <div className="pt-16">
        <AnimatePresence mode="wait">
          {/* HOME TAB */}
          {activeTab === 'home' && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {/* Hero Section */}
              <section className="relative overflow-hidden py-20 px-4">
                <div className="max-w-7xl mx-auto">
                  <div className="text-center">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.2, type: 'spring' }}
                      className="inline-block mb-6"
                    >
                      <div className="w-20 h-20 bg-gradient-to-br from-purple-500 via-pink-500 to-orange-500 rounded-2xl flex items-center justify-center mx-auto">
                        <Brain className="w-12 h-12" />
                      </div>
                    </motion.div>
                    <motion.h1
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 bg-clip-text text-transparent"
                    >
                      Know Where You Lack
                    </motion.h1>
                    <motion.p
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="text-2xl md:text-3xl text-gray-300 mb-4"
                    >
                      Intelligent, Context-Aware Learning Platform
                    </motion.p>
                    <motion.p
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 }}
                      className="text-lg md:text-xl text-gray-400 max-w-4xl mx-auto mb-8"
                    >
                      Every student learns differently — yet most education systems still follow a "one-size-fits-all"
                      approach. <span className="text-purple-400 font-semibold">Know Where You Lack</span> changes this
                      — it's an intelligent, context-aware platform that learns from the learner, not just the data.
                    </motion.p>
                    <motion.button
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.6 }}
                      onClick={() => setActiveTab('auth')}
                      className="group bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-8 py-4 rounded-full text-lg font-semibold flex items-center space-x-2 mx-auto transition-all transform hover:scale-105"
                    >
                      <span>Get Started Free</span>
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </motion.button>
                  </div>
                </div>

                {/* Floating Elements */}
                <div className="absolute top-20 left-10 w-72 h-72 bg-purple-500/10 rounded-full blur-3xl animate-pulse" />
                <div className="absolute bottom-20 right-10 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
              </section>

              {/* Problem Statement */}
              <section className="py-16 px-4 bg-slate-800/50">
                <div className="max-w-5xl mx-auto text-center">
                  <div className="inline-flex items-center space-x-2 bg-red-500/20 border border-red-500/50 px-6 py-3 rounded-full mb-6">
                    <Zap className="w-5 h-5 text-red-400" />
                    <span className="text-red-300 font-semibold">The Problem</span>
                  </div>
                  <h2 className="text-3xl md:text-4xl font-bold mb-6">
                    Students often find it hard to pinpoint{' '}
                    <span className="text-red-400">what exactly they don't understand</span>
                  </h2>
                  <p className="text-xl text-gray-300">
                    This causes repeated mistakes and low confidence, trapping learners in a cycle of frustration.
                  </p>
                </div>
              </section>

              {/* Features */}
              <section className="py-20 px-4">
                <div className="max-w-7xl mx-auto">
                  <div className="text-center mb-16">
                    <h2 className="text-4xl md:text-5xl font-bold mb-4">
                      <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                        Our Solution
                      </span>
                    </h2>
                    <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                      Know Where You Lack understands learning behavior, identifies weaknesses, and provides
                      personalized data-driven recommendations.
                    </p>
                  </div>

                  <div className="grid md:grid-cols-3 gap-8">
                    {features.map((feature, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="group relative bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-2xl p-8 hover:border-purple-500/50 transition-all"
                      >
                        <div
                          className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}
                        >
                          <feature.icon className="w-8 h-8" />
                        </div>
                        <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                        <p className="text-gray-300">{feature.description}</p>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </section>

              {/* CTA Section */}
              <section className="py-20 px-4 bg-gradient-to-r from-purple-600/20 to-pink-600/20">
                <div className="max-w-4xl mx-auto text-center">
                  <h2 className="text-4xl md:text-5xl font-bold mb-6">
                    Don't Just Learn —{' '}
                    <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                      Learn Smarter
                    </span>
                  </h2>
                  <p className="text-xl text-gray-300 mb-8">
                    Transform passive information delivery into active, insight-driven growth.
                  </p>
                  <button
                    onClick={() => setActiveTab('auth')}
                    className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-10 py-4 rounded-full text-lg font-semibold transition-all transform hover:scale-105"
                  >
                    Start Your Journey Today
                  </button>
                </div>
              </section>
            </motion.div>
          )}

          {/* ABOUT TAB */}
          {activeTab === 'about' && (
            <motion.div
              key="about"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="py-20 px-4"
            >
              <div className="max-w-7xl mx-auto">
                {/* Mission & Vision */}
                <div className="text-center mb-16">
                  <h1 className="text-5xl md:text-6xl font-bold mb-8 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                    About Us
                  </h1>
                  <div className="grid md:grid-cols-2 gap-8 mb-12">
                    <div className="bg-slate-800/50 border border-purple-500/20 rounded-2xl p-8">
                      <Target className="w-12 h-12 text-purple-400 mb-4 mx-auto" />
                      <h3 className="text-2xl font-bold mb-4 text-purple-400">Our Mission</h3>
                      <p className="text-gray-300 text-lg">
                        To transform education by empowering students to recognize their blind spots, track their
                        progress, and build confidence through targeted support.
                      </p>
                    </div>
                    <div className="bg-slate-800/50 border border-pink-500/20 rounded-2xl p-8">
                      <Zap className="w-12 h-12 text-pink-400 mb-4 mx-auto" />
                      <h3 className="text-2xl font-bold mb-4 text-pink-400">Our Vision</h3>
                      <p className="text-gray-300 text-lg">
                        In a world where education must evolve, Know Where You Lack stands as a bold step toward truly
                        individualized learning.
                      </p>
                    </div>
                  </div>
                </div>

                {/* What Makes Us Different */}
                <div className="mb-16">
                  <h2 className="text-4xl font-bold mb-8 text-center">
                    <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                      What Makes Us Different
                    </span>
                  </h2>
                  <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-orange-500/20 rounded-2xl p-8">
                    <div className="grid md:grid-cols-2 gap-6">
                      {[
                        {
                          gap: 'No topic-wise identification',
                          solution: 'Multi-granular adaptive system with topic-level analysis',
                        },
                        { gap: 'No resource recommendation', solution: 'Personalized learning materials & content matching' },
                        { gap: 'No interactive chatbot', solution: 'Conversational AI learning companion' },
                        { gap: 'Single course limitation', solution: 'Support for multiple subjects' },
                        { gap: 'No progress tracking', solution: 'Continuous tracking & analytics' },
                      ].map((item, index) => (
                        <div key={index} className="flex items-start space-x-3">
                          <CheckCircle className="w-6 h-6 text-green-400 flex-shrink-0 mt-1" />
                          <div>
                            <p className="text-red-400 line-through mb-1">{item.gap}</p>
                            <p className="text-green-400 font-semibold">✅ {item.solution}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Key Features */}
                <div className="mb-16">
                  <h2 className="text-4xl font-bold mb-8 text-center">
                    <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                      Key Features
                    </span>
                  </h2>
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {keyFeatures.map((feature, index) => (
                      <div
                        key={index}
                        className="flex items-center space-x-3 bg-slate-800/50 border border-blue-500/20 rounded-lg p-4 hover:border-blue-500/50 transition-all"
                      >
                        <CheckCircle className="w-5 h-5 text-blue-400 flex-shrink-0" />
                        <span className="text-gray-300">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Performance */}
                <div className="bg-gradient-to-br from-green-600/20 to-emerald-600/20 border border-green-500/30 rounded-2xl p-8 mb-16">
                  <h2 className="text-3xl font-bold mb-6 text-center text-green-400">Performance & Validation</h2>
                  <div className="grid md:grid-cols-3 gap-6 text-center">
                    <div>
                      <div className="text-5xl font-bold text-green-400 mb-2">87.80%</div>
                      <p className="text-gray-300">AI Course Dataset Accuracy</p>
                    </div>
                    <div>
                      <div className="text-5xl font-bold text-blue-400 mb-2">85%+</div>
                      <p className="text-gray-300">Overall Model Accuracy</p>
                    </div>
                    <div>
                      <div className="text-5xl font-bold text-purple-400 mb-2">15+</div>
                      <p className="text-gray-300">Database Tables</p>
                    </div>
                  </div>
                </div>

                {/* FAQ */}
                <div>
                  <h2 className="text-4xl font-bold mb-8 text-center">
                    <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                      Frequently Asked Questions
                    </span>
                  </h2>
                  <div className="space-y-4">
                    {faqs.map((faq, index) => (
                      <div
                        key={index}
                        className="bg-slate-800/50 border border-purple-500/20 rounded-xl p-6 hover:border-purple-500/50 transition-all"
                      >
                        <h3 className="text-xl font-bold text-purple-400 mb-3">{faq.q}</h3>
                        <p className="text-gray-300">{faq.a}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* AUTH TAB */}
          {activeTab === 'auth' && (
            <motion.div
              key="auth"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="py-20 px-4"
            >
              <div className="max-w-md mx-auto">
                <Login onLoginSuccess={onLoginSuccess} />
              </div>
            </motion.div>
          )}

          {/* CONTACT TAB */}
          {activeTab === 'contact' && (
            <motion.div
              key="contact"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="py-20 px-4"
            >
              <div className="max-w-5xl mx-auto">
                <div className="text-center mb-12">
                  <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                    Contact Us
                  </h1>
                  <p className="text-xl text-gray-300">We'd love to hear from you!</p>
                </div>

                <div className="grid md:grid-cols-2 gap-8">
                  {/* Contact Form */}
                  <div className="bg-slate-800/50 border border-purple-500/20 rounded-2xl p-8">
                    <h2 className="text-2xl font-bold mb-6">Send us a message</h2>
                    <form onSubmit={handleContact} className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Name</label>
                        <input
                          type="text"
                          value={contactForm.name}
                          onChange={(e) => setContactForm({ ...contactForm, name: e.target.value })}
                          className="w-full bg-slate-900/50 border border-purple-500/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-500"
                          placeholder="Your name"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                        <input
                          type="email"
                          value={contactForm.email}
                          onChange={(e) => setContactForm({ ...contactForm, email: e.target.value })}
                          className="w-full bg-slate-900/50 border border-purple-500/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-500"
                          placeholder="your@email.com"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Phone (Optional)</label>
                        <input
                          type="tel"
                          value={contactForm.phone}
                          onChange={(e) => setContactForm({ ...contactForm, phone: e.target.value })}
                          className="w-full bg-slate-900/50 border border-purple-500/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-500"
                          placeholder="+91 1234567890"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Subject</label>
                        <input
                          type="text"
                          value={contactForm.subject}
                          onChange={(e) => setContactForm({ ...contactForm, subject: e.target.value })}
                          className="w-full bg-slate-900/50 border border-purple-500/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-500"
                          placeholder="How can we help?"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Message</label>
                        <textarea
                          value={contactForm.message}
                          onChange={(e) => setContactForm({ ...contactForm, message: e.target.value })}
                          className="w-full bg-slate-900/50 border border-purple-500/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-500 h-32 resize-none"
                          placeholder="Tell us more..."
                          required
                        />
                      </div>
                      <button
                        type="submit"
                        className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white py-3 rounded-lg font-semibold transition-all transform hover:scale-105 flex items-center justify-center space-x-2"
                      >
                        <span>Send Message</span>
                        <Send className="w-5 h-5" />
                      </button>
                    </form>
                  </div>

                  {/* Contact Info */}
                  <div className="space-y-6">
                    <div className="bg-slate-800/50 border border-purple-500/20 rounded-2xl p-8">
                      <h2 className="text-2xl font-bold mb-6">Contact Information</h2>
                      <div className="space-y-4">
                        <div className="flex items-start space-x-3">
                          <Mail className="w-6 h-6 text-purple-400 flex-shrink-0 mt-1" />
                          <div>
                            <p className="font-semibold text-gray-300">Email</p>
                            <p className="text-gray-400">support@knowwhereyoulack.com</p>
                            <p className="text-gray-400">help@knowwhereyoulack.com</p>
                          </div>
                        </div>
                        <div className="flex items-start space-x-3">
                          <Clock className="w-6 h-6 text-pink-400 flex-shrink-0 mt-1" />
                          <div>
                            <p className="font-semibold text-gray-300">Office Hours</p>
                            <p className="text-gray-400">Mon-Fri, 10 AM - 6 PM IST</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-800/50 border border-pink-500/20 rounded-2xl p-8">
                      <h3 className="text-xl font-bold mb-4">Team Members</h3>
                      <ul className="space-y-2 text-gray-300">
                        <li>• B. Komala Manaswini (2410030104)</li>
                        <li>• Kulkarni Sahithi (2410030057)</li>
                        <li>• Aarushi Chakraborty (2410030008)</li>
                        <li>• P. Lalitha Preethi (2410030103)</li>
                        <li>• Joshika Mannam (2410030108)</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <footer className="bg-slate-900/80 border-t border-purple-500/20 py-8 px-4">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-gray-400 mb-2">
            © 2025 Know Where You Lack. All rights reserved.
          </p>
          <p className="text-sm text-gray-500">
            "Your Learning, Intelligently Adapted"
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
