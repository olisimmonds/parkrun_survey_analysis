import type { ChatMessage, QueryConfig } from '@/types/chat';
import { getRandomResponse } from '@/data/mock/chatResponses';
import { generateId, sleep } from '@/lib/utils';

export const chatService = {
  async sendMessage(
    content: string,
    config: QueryConfig,
    onChunk?: (chunk: string) => void,
  ): Promise<ChatMessage> {
    const delay = config.mode === 'deep-research' ? 3500 : 1800;
    await sleep(delay);

    const { content: responseContent, sources } = getRandomResponse();

    // Simulate streaming chunks
    if (onChunk) {
      const words = responseContent.split(' ');
      for (const word of words) {
        await sleep(30);
        onChunk(word + ' ');
      }
    }

    return {
      id: generateId(),
      role: 'assistant',
      content: responseContent,
      timestamp: new Date().toISOString(),
      sources,
    };
  },
};
