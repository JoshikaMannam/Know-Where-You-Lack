// API Configuration
// Automatically uses environment-based URL
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8082/api';

// Export configured axios instance if needed
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests automatically
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('jwt_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
