import type { ChatMessage, QueryConfig, Source } from '@/types/chat';
import { defaultConfig } from '@/types/config';
import { getRandomResponse } from '@/data/mock/chatResponses';
import { generateId, sleep } from '@/lib/utils';

/** Parse a Server-Sent Events stream and yield typed events. */
async function* parseSSE(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<{ event: string; data: string }> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          yield { event: currentEvent, data: line.slice(6) };
          currentEvent = 'message';
        }
        // blank line resets event type (standard SSE)
        else if (line === '') {
          currentEvent = 'message';
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export const chatService = {
  async sendMessage(
    content: string,
    config: QueryConfig,
    onChunk?: (chunk: string) => void,
  ): Promise<ChatMessage> {
    if (defaultConfig.mockMode) {
      const delay = config.mode === 'deep-research' ? 3500 : 1800;
      await sleep(delay);
      const { content: responseContent, sources } = getRandomResponse();
      if (onChunk) {
        for (const word of responseContent.split(' ')) {
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
    }

    // ── Real SSE streaming ─────────────────────────────────────────────────
    const res = await fetch(`${defaultConfig.apiBaseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: content,
        config: {
          mode: config.mode,
          datasetIds: config.datasetIds,
          maxSources: config.maxSources,
        },
      }),
    });

    if (!res.ok || !res.body) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(
        (detail as { detail?: string }).detail ?? `Chat API error ${res.status}`,
      );
    }

    let fullContent = '';
    let sources: Source[] = [];

    for await (const { event, data } of parseSSE(res.body)) {
      switch (event) {
        case 'chunk': {
          const parsed = JSON.parse(data) as { text: string };
          fullContent += parsed.text;
          onChunk?.(parsed.text);
          break;
        }
        case 'sources': {
          sources = JSON.parse(data) as Source[];
          break;
        }
        case 'error': {
          const parsed = JSON.parse(data) as { message: string };
          throw new Error(parsed.message);
        }
        case 'done':
        case 'status':
        default:
          break;
      }
    }

    return {
      id: generateId(),
      role: 'assistant',
      content: fullContent,
      timestamp: new Date().toISOString(),
      sources,
    };
  },
};
