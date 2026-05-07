export type MessageRole = 'user' | 'assistant';
export type QueryMode = 'standard' | 'deep-research';

export interface Source {
  id: string;
  name: string;
  excerpt: string;
  datasetId: string;
  relevanceScore: number;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  sources?: Source[];
  isLoading?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  mode: QueryMode;
  createdAt: string;
}

export interface QueryConfig {
  mode: QueryMode;
  datasetIds: string[];
  maxSources: number;
}
