import { useState, useEffect } from 'react';
import { BookOpen, MessageSquare, BarChart3, Plus, User, LogOut, ChevronRight, Check, X, Send, FileText } from 'lucide-react';
import '../styles/Chatbot.css';
import '../styles/Quiz.css';
import '../styles/Dashboard.css';
import '../styles/Notes.css';
import { API_BASE_URL } from '../config/api';

interface ApiError {
  message: string;
  status: number;
}

const api = {
  get: async <T,>(url: string): Promise<T> => {
    const response = await fetch(`${API_BASE_URL}${url}`, {
      headers: { 'Authorization': `Bearer ${localStorage.getItem('token') || ''}` }
    });
    if (!response.ok) {
      const error = await response.json() as ApiError;
      throw new Error(error.message || 'Request failed');
    }
    return response.json();
  },
  post: async <T,>(url: string, data: unknown): Promise<T> => {
    const response = await fetch(`${API_BASE_URL}${url}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
      },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      const error = await response.json() as ApiError;
      throw new Error(error.message || 'Request failed');
    }
    return response.json();
  }
};

// Format chatbot response to have proper paragraphs and bullets
const formatMessage = (text: string): string => {
  // Split by double newlines for paragraphs
  const formatted = text.split('\n\n').map(para => {
    para = para.trim();
    
    // Check if it's a bullet point list
    if (para.includes('\n-') || para.includes('\n•') || para.includes('\n*')) {
      const lines = para.split('\n');
      let html = '<ul>';
      lines.forEach(line => {
        const trimmed = line.trim();
        if (trimmed.startsWith('-') || trimmed.startsWith('•') || trimmed.startsWith('*')) {
          html += `<li>${trimmed.substring(1).trim()}</li>`;
        } else if (trimmed) {
          html += `<li>${trimmed}</li>`;
        }
      });
      html += '</ul>';
      return html;
    }
    
    // Check if it's a numbered list
    if (/^\d+\./.test(para)) {
      const lines = para.split('\n');
      let html = '<ol>';
      lines.forEach(line => {
        const trimmed = line.trim();
        if (/^\d+\./.test(trimmed)) {
          html += `<li>${trimmed.replace(/^\d+\.\s*/, '')}</li>`;
        }
      });
      html += '</ol>';
      return html;
    }
    
    // Regular paragraph
    return `<p>${para}</p>`;
  }).join('');
  
  return formatted;
};

export default function QuizLearningApp() {
  type View = 'login' | 'dashboard' | 'quiz' | 'chat' | 'analysis' | 'chatbot' | 'analytics' | 'results' | 'notes';

  interface User {
    id: number;
    email: string;
    fullName: string;
  }

  interface Topic {
    id: number;
    name: string;
    description: string;
    quizCount?: number;
  }

  interface QuizAnswer {
    questionId: number;
    correct: boolean;
    selected: number;
  }

  interface QuizQuestion {
    id: number;
    question: string;
    options: string[];
    correctAnswer: number;
  }

  interface Quiz {
    id: number;
    title: string;
    questions: QuizQuestion[];
  }

  interface ChatMessage {
    id: number;
    text: string;
    sender: 'user' | 'bot' | 'ai';
    timestamp: Date;
  }

  interface Note {
    id: number;
    title: string;
    content: string;
    subject: string;
    createdAt: string;
    updatedAt: string;
  }

  const [currentView, setCurrentView] = useState<View>('login');
  const [user, setUser] = useState<User | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [score, setScore] = useState(0);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [quizAnswers, setQuizAnswers] = useState<QuizAnswer[]>([]);
  const [isTyping, setIsTyping] = useState(false);

  const LoginView = () => {
    const [credentials, setCredentials] = useState({
      email: '',
      password: '',
      fullName: ''
    });
    const [isSignup, setIsSignup] = useState(false);

    const handleAuth = async () => {
      setLoading(true);
      try {
        const endpoint = isSignup ? '/auth/register' : '/auth/login';
        const response = await api.post<any>(endpoint, credentials);

        if (isSignup) {
          // register -> backend returns user object directly
          const userResp = response as { userId?: number; name?: string; email?: string };
          setUser({ id: userResp.userId || 0, email: userResp.email || credentials.email, fullName: userResp.name || credentials.fullName });
          setCurrentView('dashboard');
        } else {
          // login -> backend returns { token }
          const loginResp = response as { token?: string };
          if (!loginResp?.token) throw new Error('Authentication failed: no token returned');
          localStorage.setItem('token', loginResp.token);

          // fetch user profile
          try {
            const me = await api.get<{ userId: number; name: string; email: string }>('/auth/me');
            setUser({ id: me.userId, email: me.email, fullName: me.name });
          } catch {
            // fallback demo user
            setUser({ id: 1, email: credentials.email || 'demo@example.com', fullName: credentials.fullName || 'Demo User' });
          }

          setCurrentView('dashboard');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Authentication failed';
        alert(message);
        if (isSignup) return;
        setUser({
          id: 1,
          email: credentials.email || 'demo@example.com',
          fullName: credentials.fullName || 'Demo User'
        });
        setCurrentView('dashboard');
      } finally {
        setLoading(false);
      }
    };

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-indigo-600 rounded-full mb-4">
              <BookOpen className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900">AI Quiz Master</h1>
            <p className="text-gray-600 mt-2">Learn smarter with personalized quizzes</p>
          </div>

          <div className="space-y-4">
            {isSignup && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                <input
                  type="text"
                  value={credentials.fullName}
                  onChange={(e) => setCredentials({ ...credentials, fullName: e.target.value })}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="Enter your full name"
                  required
                />
              </div>
            )}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
              <input
                type="email"
                value={credentials.email}
                onChange={(e) => setCredentials({ ...credentials, email: e.target.value })}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="Enter email"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
              <input
                type="password"
                value={credentials.password}
                onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="Enter password"
                required
              />
            </div>
            <button
              onClick={handleAuth}
              disabled={loading}
              className="w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors disabled:opacity-50"
            >
              {loading ? 'Loading...' : (isSignup ? 'Sign Up' : 'Login')}
            </button>
          </div>

          <div className="mt-6 text-center">
            <button
              onClick={() => setIsSignup(!isSignup)}
              className="text-indigo-600 hover:text-indigo-700 font-medium"
            >
              {isSignup ? 'Already have an account? Login' : "Don't have an account? Sign Up"}
            </button>
          </div>
        </div>
      </div>
    );
  };

  const DashboardView = () => {
    const [showAddTopic, setShowAddTopic] = useState(false);
    const [newTopic, setNewTopic] = useState({ name: '', description: '' });

    useEffect(() => { loadTopics(); }, []);

    const loadTopics = async () => {
      try {
        console.log('🔍 Loading topics from backend...');
        // Call backend endpoint: GET /api/quiz/topics
        const response = await api.get<Array<{
          topicId: number;
          topicName: string;
          difficultyLevel: string;
          description: string;
          questionCount: number;
        }>>('/quiz/topics');
        
        console.log(`✅ Received ${response.length} topics from backend`);
        
        // Transform backend response to frontend Topic format
        const transformedTopics = response.map(t => ({
          id: t.topicId,
          name: t.topicName,
          description: t.description || 'No description available',
          quizCount: t.questionCount
        }));
        
        setTopics(transformedTopics);
        console.log(`📝 Loaded topics:`, transformedTopics);
        
      } catch (error) {
        console.error('❌ Error loading topics:', error);
        // Fallback to demo topics
        setTopics([
          { id: 1, name: 'JavaScript Fundamentals', description: 'Learn the basics of JavaScript', quizCount: 5 },
          { id: 2, name: 'React Hooks', description: 'Master React Hooks', quizCount: 3 },
          { id: 3, name: 'Data Structures', description: 'Essential data structures', quizCount: 7 }
        ]);
      }
    };

    const handleAddTopic = async () => {
      try {
        await api.post('/topics', newTopic);
        loadTopics();
        setShowAddTopic(false);
        setNewTopic({ name: '', description: '' });
      } catch {
        alert('Failed to add topic. Try again.');
      }
    };

    const startQuiz = (topic: Topic) => {
      setCurrentView('quiz');
      loadQuiz(topic.id);
    };

    const loadQuiz = async (topicId: number) => {
      try {
        console.log(`🔍 Loading 10 questions for topic ${topicId}...`);
        
        // Call backend endpoint: /api/quiz/{topicId}?questionCount=10
        const data = await api.get<Array<{
          id: number;
          questionText: string;
          options: string[];
        }>>(`/quiz/${topicId}?questionCount=10`);
        
        console.log(`✅ Received ${data.length} questions from backend`);
        
        // Transform backend response to frontend Quiz format
        const transformedQuiz: Quiz = {
          id: topicId,
          title: topics.find(t => t.id === topicId)?.name || 'Quiz',
          questions: data.map((q) => ({
            id: q.id,
            question: q.questionText,
            options: q.options,
            correctAnswer: 0
          }))
        };
        
        console.log(`📝 Quiz loaded with ${transformedQuiz.questions.length} questions`);
        setQuiz(transformedQuiz);
        
      } catch (error) {
        console.error('❌ Error loading quiz:', error);
        setQuiz({
          id: 1,
          title: topics.find(t => t.id === topicId)?.name || 'Quiz',
          questions: [
            {
              id: 1,
              question: 'Error loading questions from database.',
              options: [
                'Please check backend connection',
                'Database might be empty',
                'Backend server not running',
                'Network error occurred'
              ],
              correctAnswer: 0
            }
          ]
        });
      }
    };

    return (
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <BookOpen className="w-8 h-8 text-indigo-600" />
              <h1 className="text-2xl font-bold text-gray-900">AI Quiz Master</h1>
            </div>
            <div className="flex items-center space-x-6">
              <button onClick={() => setCurrentView('chatbot')} className="flex items-center space-x-2 text-gray-700 hover:text-indigo-600 transition-colors">
                <MessageSquare className="w-5 h-5" />
                <span>AI Tutor</span>
              </button>
              <button onClick={() => setCurrentView('notes')} className="flex items-center space-x-2 text-gray-700 hover:text-indigo-600 transition-colors">
                <FileText className="w-5 h-5" />
                <span>Notes</span>
              </button>
              <button onClick={() => setCurrentView('analytics')} className="flex items-center space-x-2 text-gray-700 hover:text-indigo-600 transition-colors">
                <BarChart3 className="w-5 h-5" />
                <span>Analytics</span>
              </button>
              <div className="flex items-center space-x-3 pl-6 border-l border-gray-300">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center">
                    <User className="w-5 h-5 text-white" />
                  </div>
                  <span className="font-medium text-gray-700">{user?.fullName}</span>
                </div>
                <button onClick={() => { setUser(null); setCurrentView('login'); }} className="text-gray-500 hover:text-red-600 transition-colors">
                  <LogOut className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </nav>

        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex justify-between items-center mb-8">
            <div>
              <h2 className="text-3xl font-bold text-gray-900">My Topics</h2>
              <p className="text-gray-600 mt-2">Choose a topic to start learning</p>
            </div>
            <button onClick={() => setShowAddTopic(true)} className="flex items-center space-x-2 bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors">
              <Plus className="w-5 h-5" />
              <span>Add Topic</span>
            </button>
          </div>

          {showAddTopic && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
              <div className="bg-white rounded-xl p-6 w-full max-w-md">
                <h3 className="text-xl font-bold mb-4">Add New Topic</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Topic Name</label>
                    <input type="text" value={newTopic.name} onChange={(e) => setNewTopic({ ...newTopic, name: e.target.value })} className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
                    <textarea value={newTopic.description} onChange={(e) => setNewTopic({ ...newTopic, description: e.target.value })} className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent" rows={3} />
                  </div>
                  <div className="flex space-x-3">
                    <button onClick={handleAddTopic} className="flex-1 bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700">Add Topic</button>
                    <button onClick={() => setShowAddTopic(false)} className="flex-1 bg-gray-200 text-gray-700 py-2 rounded-lg hover:bg-gray-300">Cancel</button>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {topics.map((topic) => (
              <div key={topic.id} className="bg-white rounded-xl shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-900 mb-2">{topic.name}</h3>
                <p className="text-gray-600 mb-4">{topic.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">{topic.quizCount || 0} questions</span>
                  <button onClick={() => startQuiz(topic)} className="flex items-center space-x-2 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors">
                    <span>Start Quiz (10 Questions)</span>
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const QuizView = () => {
    const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
    const [showResult, setShowResult] = useState(false);

    const handleAnswerSelect = (index: number) => {
      if (!showResult) setSelectedAnswer(index);
    };

    const handleSubmitAnswer = () => {
      if (selectedAnswer === null) return;
      if (!quiz) return;
      const currentQ = quiz.questions[currentQuestion];
      const isCorrect = selectedAnswer === currentQ.correctAnswer;

      setQuizAnswers((prev) => [...prev, { questionId: currentQ.id, correct: isCorrect, selected: selectedAnswer }]);
      if (isCorrect) setScore(score + 1);
      setShowResult(true);
    };

    const handleNextQuestion = () => {
      if (!quiz) return;
      if (currentQuestion < quiz.questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1);
        setSelectedAnswer(null);
        setShowResult(false);
      } else {
        submitQuizResults();
      }
    };

    const submitQuizResults = async () => {
      if (!quiz) return;
      try {
        await api.post('/quiz-results', {
          quizId: quiz.id,
          score,
          totalQuestions: quiz.questions.length,
          answers: quizAnswers
        });
      } catch { }
      setCurrentView('results');
    };

    if (!quiz) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-600 mx-auto" />
            <p className="mt-4 text-gray-600">Loading quiz...</p>
          </div>
        </div>
      );
    }

    const question = quiz.questions[currentQuestion];

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
        <div className="max-w-3xl mx-auto pt-8">
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex justify-between items-center mb-6">
              <button onClick={() => { setCurrentView('dashboard'); setQuiz(null); setCurrentQuestion(0); setScore(0); setQuizAnswers([]); setSelectedAnswer(null); setShowResult(false); }} className="text-gray-600 hover:text-gray-900 transition-colors">← Back to Dashboard</button>
              <div className="flex items-center space-x-4">
                <span className="text-sm font-medium text-gray-600">Question {currentQuestion + 1} of {quiz.questions.length}</span>
                <span className="bg-indigo-100 text-indigo-700 px-4 py-2 rounded-full font-semibold">Score: {score}/{quiz.questions.length}</span>
              </div>
            </div>

            <div className="mb-8">
              <div className="w-full bg-gray-200 rounded-full h-2 mb-6">
                <div className="bg-indigo-600 h-2 rounded-full transition-all duration-300" style={{ width: `${((currentQuestion + 1) / quiz.questions.length) * 100}%` }}></div>
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">{question.question}</h2>
            </div>

            <div className="space-y-4 mb-8">
              {question.options.map((option, index) => (
                <button
                  key={index}
                  onClick={() => handleAnswerSelect(index)}
                  disabled={showResult}
                  className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                    selectedAnswer === index
                      ? showResult
                        ? index === question.correctAnswer
                          ? 'border-green-500 bg-green-50'
                          : 'border-red-500 bg-red-50'
                        : 'border-indigo-500 bg-indigo-50'
                      : showResult && index === question.correctAnswer
                      ? 'border-green-500 bg-green-50'
                      : 'border-gray-300 hover:border-indigo-300 bg-white'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{option}</span>
                    {showResult && (
                      <span>
                        {index === question.correctAnswer ? (
                          <Check className="w-6 h-6 text-green-600" />
                        ) : selectedAnswer === index ? (
                          <X className="w-6 h-6 text-red-600" />
                        ) : null}
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>

            <div className="flex justify-end">
              {!showResult ? (
                <button
                  onClick={handleSubmitAnswer}
                  disabled={selectedAnswer === null}
                  className="bg-indigo-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Submit Answer
                </button>
              ) : (
                <button
                  onClick={handleNextQuestion}
                  className="bg-indigo-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors"
                >
                  {currentQuestion < quiz.questions.length - 1 ? 'Next Question' : 'Finish Quiz'}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const ResultsView = () => {
    const percentage = quiz ? Math.round((score / quiz.questions.length) * 100) : 0;

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center">
          <div className={`w-24 h-24 rounded-full mx-auto mb-6 flex items-center justify-center ${percentage >= 70 ? 'bg-green-100' : 'bg-yellow-100'}`}>
            <span className="text-4xl">{percentage >= 70 ? '🎉' : '📚'}</span>
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Quiz Complete!</h2>
          <p className="text-gray-600 mb-8">You scored</p>
          <div className="text-6xl font-bold text-indigo-600 mb-8">
            {score}/{quiz?.questions.length || 0}
          </div>
          <p className="text-xl text-gray-700 mb-8">{percentage}% Correct</p>
          <div className="space-y-3">
            <button onClick={() => { setCurrentView('dashboard'); setQuiz(null); setCurrentQuestion(0); setScore(0); setQuizAnswers([]); }} className="w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors">Back to Dashboard</button>
            <button onClick={() => setCurrentView('chatbot')} className="w-full bg-gray-200 text-gray-700 py-3 rounded-lg font-semibold hover:bg-gray-300 transition-colors">Ask AI Tutor for Help</button>
          </div>
        </div>
      </div>
    );
  };

  const ChatbotView = () => {
    // Load chat history when component mounts
    useEffect(() => {
      const loadChatHistory = async () => {
        try {
          const response = await api.get('/chat/history');
          const history = response as Array<{ text: string; sender: string; timestamp: string }>;
          const messages: ChatMessage[] = history.map((msg, index) => ({
            id: index,
            text: msg.text,
            sender: msg.sender as 'user' | 'ai',
            timestamp: new Date(msg.timestamp)
          }));
          setChatMessages(messages);
        } catch (error) {
          console.log('No chat history or failed to load');
        }
      };
      loadChatHistory();
    }, []);

    const sendMessage = async () => {
      if (!chatInput.trim()) return;

      const userMessage: ChatMessage = { 
        id: Date.now(), 
        text: chatInput, 
        sender: 'user', 
        timestamp: new Date() 
      };
      setChatMessages(prev => [...prev, userMessage]);
      const currentInput = chatInput;
      setChatInput('');
      setIsTyping(true);

      try {
        // Send to new chat controller
        const response = await api.post('/chat/message', { message: currentInput });
        const botResponse = response as { response: string };
        
        const botMessage: ChatMessage = { 
          id: Date.now() + 1, 
          text: botResponse.response || "I'm having trouble responding right now.", 
          sender: 'ai', 
          timestamp: new Date() 
        };
        setChatMessages(prev => [...prev, botMessage]);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Network error';
        const botMessage: ChatMessage = {
          id: Date.now() + 1,
          text: `⚠️ ${errorMessage}\n\nI'm here to help with:\n• Mathematics & Science\n• Programming & Computer Science\n• Language & Literature\n• Study strategies & tips\n\nWhat would you like to learn about?`,
          sender: 'ai',
          timestamp: new Date()
        };
        setChatMessages(prev => [...prev, botMessage]);
      } finally {
        setIsTyping(false);
      }
    };

    return (
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-5xl mx-auto px-4 py-4 flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <MessageSquare className="w-8 h-8 text-indigo-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Skilli - AI Tutor</h1>
                <p className="text-xs text-gray-500">Education-focused assistant</p>
              </div>
            </div>
            <button onClick={() => setCurrentView('dashboard')} className="text-gray-600 hover:text-gray-900 transition-colors">← Back to Dashboard</button>
          </div>
        </nav>

        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="bg-white rounded-2xl shadow-lg h-[600px] flex flex-col">
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {chatMessages.length === 0 && (
                <div className="text-center text-gray-500 mt-20">
                  <MessageSquare className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                  <p className="text-lg font-semibold">👋 Hi! I'm Skilli, your AI study companion!</p>
                  <p className="text-sm mt-2">I can help you with:</p>
                  <div className="mt-4 space-y-2 text-left max-w-md mx-auto">
                    <div className="flex items-start space-x-2">
                      <span className="text-indigo-600">📚</span>
                      <span className="text-sm">Math, Science, Programming & more</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span className="text-indigo-600">💡</span>
                      <span className="text-sm">Explaining complex concepts simply</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span className="text-indigo-600">✍️</span>
                      <span className="text-sm">Study tips and learning strategies</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span className="text-indigo-600">🎯</span>
                      <span className="text-sm">Practice problems and examples</span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 mt-6">Ask me anything educational!</p>
                </div>
              )}
              {chatMessages.map((msg, index) => (
                <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                    msg.sender === 'user' 
                      ? 'bg-indigo-600 text-white' 
                      : 'bg-gray-100 text-gray-900 border border-gray-200'
                  }`}>
                    {msg.sender === 'ai' && (
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-xs font-semibold text-indigo-600">🤖 Skilli</span>
                      </div>
                    )}
                    {msg.sender === 'ai' ? (
                      <div 
                        className="message-content"
                        dangerouslySetInnerHTML={{ __html: formatMessage(msg.text) }}
                      />
                    ) : (
                      <p className="whitespace-pre-wrap">{msg.text}</p>
                    )}
                    <span className="text-xs opacity-70 mt-1 block">
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 px-4 py-3 rounded-2xl border border-gray-200">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-gray-200 p-4 bg-gray-50">
              <div className="flex space-x-3">
                <input 
                  type="text" 
                  value={chatInput} 
                  onChange={(e) => setChatInput(e.target.value)} 
                  onKeyPress={(e) => e.key === 'Enter' && !isTyping && sendMessage()} 
                  placeholder="Ask about math, science, programming..." 
                  disabled={isTyping}
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50" 
                />
                <button 
                  onClick={sendMessage} 
                  disabled={isTyping || !chatInput.trim()}
                  className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-2 text-center">
                💡 Tip: Ask specific questions for better answers
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const NotesView = () => {
    const [notes, setNotes] = useState<Note[]>([]);
    const [notesLoading, setNotesLoading] = useState(true);
    const [notesError, setNotesError] = useState('');

    // Load notes from backend on component mount
    useEffect(() => {
      const fetchNotes = async () => {
        try {
          setNotesLoading(true);
          const response = await api.get('/notes');
          const notesData = response as Note[];
          setNotes(notesData);
          setNotesError('');
        } catch (err) {
          console.error('Failed to fetch notes:', err);
          setNotesError('Failed to load notes. Please try again.');
        } finally {
          setNotesLoading(false);
        }
      };
      fetchNotes();
    }, []);

    return (
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <FileText className="w-8 h-8 text-indigo-600" />
              <h1 className="text-2xl font-bold text-gray-900">My Notes</h1>
            </div>
            <button onClick={() => setCurrentView('dashboard')} className="text-gray-600 hover:text-gray-900 transition-colors">← Back to Dashboard</button>
          </div>
        </nav>

        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Study Notes</h2>
              <p className="text-gray-600 mt-1">Your learning materials from backend</p>
            </div>
            <span className="status-badge status-badge--success">
              ✓ Loaded from backend
            </span>
          </div>

          {notesLoading && (
            <div className="loading text-center py-12">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
              <p className="mt-4 text-gray-600">Loading notes...</p>
            </div>
          )}

          {notesError && (
            <div className="error text-center py-12">
              <p className="text-red-600 text-lg">{notesError}</p>
            </div>
          )}

          {!notesLoading && !notesError && (
            <div className="notes-grid">
              {notes.map(note => (
                <div key={note.id} className="note-card">
                  <div className="note-category">{note.subject}</div>
                  <h3 className="note-title">{note.title}</h3>
                  <p className="note-content">{note.content}</p>
                  <div className="note-footer">
                    <span className="note-date">
                      {new Date(note.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  const AnalyticsView = () => {
    return (
      <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <BarChart3 className="w-8 h-8 text-indigo-600" />
            <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
          </div>
          <button onClick={() => setCurrentView('dashboard')} className="text-gray-600 hover:text-gray-900 transition-colors">← Back to Dashboard</button>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="dashboard-stats">
          <div className="stat-card stat-card--streak">
            <div className="stat-icon">🔥</div>
            <div className="stat-content">
              <div className="stat-value">5</div>
              <div className="stat-label">Day Streak</div>
            </div>
          </div>
          <div className="stat-card stat-card--quizzes">
            <div className="stat-icon">📝</div>
            <div className="stat-content">
              <div className="stat-value">12</div>
              <div className="stat-label">Quizzes Completed</div>
            </div>
          </div>
          <div className="stat-card stat-card--accuracy">
            <div className="stat-icon">🎯</div>
            <div className="stat-content">
              <div className="stat-value">78%</div>
              <div className="stat-label">Average Accuracy</div>
            </div>
          </div>
          <div className="stat-card stat-card--weak">
            <div className="stat-icon">📚</div>
            <div className="stat-content">
              <div className="stat-value">3</div>
              <div className="stat-label">Weak Subjects</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Recent Performance</h3>
          <div className="space-y-4">
            {['JavaScript Fundamentals', 'React Hooks', 'Data Structures'].map((topic, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="font-medium text-gray-700">{topic}</span>
                <div className="flex items-center space-x-4">
                  <div className="w-48 bg-gray-200 rounded-full h-2">
                    <div className="bg-indigo-600 h-2 rounded-full" style={{ width: `${70 + index * 10}%` }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-600">{70 + index * 10}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
    );
  };

  return (
    <div>
      {currentView === 'login' && <LoginView />}
      {currentView === 'dashboard' && <DashboardView />}
      {currentView === 'quiz' && <QuizView />}
      {currentView === 'results' && <ResultsView />}
      {currentView === 'chatbot' && <ChatbotView />}
      {currentView === 'notes' && <NotesView />}
      {currentView === 'analytics' && <AnalyticsView />}
    </div>
  );
}
