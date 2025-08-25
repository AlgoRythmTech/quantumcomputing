import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  sources?: any[];
  pdfUrl?: string;
}

interface ChatState {
  messages: Message[];
  isTyping: boolean;
  searchWeb: boolean;
  generatePdf: boolean;
  conversationId: string | null;
}

const initialState: ChatState = {
  messages: [],
  isTyping: false,
  searchWeb: false,
  generatePdf: false,
  conversationId: null,
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    addMessage: (state, action: PayloadAction<Message>) => {
      state.messages.push(action.payload);
    },
    setTyping: (state, action: PayloadAction<boolean>) => {
      state.isTyping = action.payload;
    },
    setSearchWeb: (state, action: PayloadAction<boolean>) => {
      state.searchWeb = action.payload;
    },
    setGeneratePdf: (state, action: PayloadAction<boolean>) => {
      state.generatePdf = action.payload;
    },
    setConversationId: (state, action: PayloadAction<string>) => {
      state.conversationId = action.payload;
    },
    clearMessages: (state) => {
      state.messages = [];
      state.conversationId = null;
    },
  },
});

export const {
  addMessage,
  setTyping,
  setSearchWeb,
  setGeneratePdf,
  setConversationId,
  clearMessages,
} = chatSlice.actions;

export default chatSlice.reducer;
