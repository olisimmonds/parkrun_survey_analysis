'use client';

import { create } from 'zustand';
import type { ChatMessage, ChatSession, QueryConfig, QueryMode } from '@/types/chat';
import { chatService } from '@/services/chat.service';
import { generateId } from '@/lib/utils';

interface ChatState {
  session: ChatSession;
  isSending: boolean;
  config: QueryConfig;
  setMode: (mode: QueryMode) => void;
  sendMessage: (content: string) => Promise<void>;
  clearSession: () => void;
}

function createSession(): ChatSession {
  return {
    id: generateId(),
    title: 'New conversation',
    messages: [],
    mode: 'standard',
    createdAt: new Date().toISOString(),
  };
}

export const useChatStore = create<ChatState>((set, get) => ({
  session: createSession(),
  isSending: false,
  config: {
    mode: 'standard',
    datasetIds: [],
    maxSources: 3,
  },

  setMode: (mode) =>
    set((state) => ({
      config: { ...state.config, mode },
      session: { ...state.session, mode },
    })),

  sendMessage: async (content) => {
    const { config, session } = get();

    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };

    const loadingId = generateId();
    const loadingMessage: ChatMessage = {
      id: loadingId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isLoading: true,
    };

    set((state) => ({
      isSending: true,
      session: {
        ...state.session,
        messages: [...state.session.messages, userMessage, loadingMessage],
        title:
          state.session.messages.length === 0
            ? content.slice(0, 50)
            : state.session.title,
      },
    }));

    try {
      // onChunk: update the assistant bubble in real-time as tokens arrive.
      const response = await chatService.sendMessage(content, config, (chunk) => {
        set((state) => ({
          session: {
            ...state.session,
            messages: state.session.messages.map((m) =>
              m.id === loadingId
                ? { ...m, content: m.content + chunk, isLoading: false }
                : m,
            ),
          },
        }));
      });

      // Final update attaches sources (and ensures content is canonical).
      set((state) => ({
        isSending: false,
        session: {
          ...state.session,
          messages: state.session.messages.map((m) =>
            m.id === loadingId
              ? { ...response, id: loadingId, isLoading: false }
              : m,
          ),
        },
      }));
    } catch {
      set((state) => ({
        isSending: false,
        session: {
          ...state.session,
          messages: state.session.messages.map((m) =>
            m.id === loadingId
              ? {
                  ...m,
                  isLoading: false,
                  content: 'Something went wrong. Please try again.',
                }
              : m,
          ),
        },
      }));
    }
  },

  clearSession: () => set({ session: createSession() }),
}));
