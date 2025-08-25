import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Chat API
export const chatAPI = {
  sendMessage: async (content: string, userName: string = 'User') => {
    const response = await api.post('/chat', {
      content,
      user_name: userName
    });
    return response.data;
  },
};

// Image Analysis API
export const imageAPI = {
  analyzeImage: async (file: File, question?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (question) {
      formData.append('question', question);
    }
    
    const response = await api.post('/api/analyze-image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

// Web Search API
export const searchAPI = {
  search: async (query: string, numResults: number = 5) => {
    const response = await api.post('/api/search', {
      query,
      num_results: numResults,
    });
    return response.data;
  },
};

// File Upload API
export const fileAPI = {
  upload: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  download: (filename: string) => {
    return `${API_BASE_URL}/api/download/${filename}`;
  },
};

export default api;
