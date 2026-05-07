'use client';

import { BookOpen, Loader2 } from 'lucide-react';
import type { ChatMessage as ChatMessageType } from '@/types/chat';
import { cn } from '@/lib/utils';

interface Props {
  message: ChatMessageType;
}

function renderMarkdown(text: string) {
  // Very lightweight Markdown rendering for bold, bullets, and blockquotes
  return text
    .split('\n')
    .map((line, i) => {
      if (line.startsWith('> ')) {
        return (
          <blockquote
            key={i}
            className="my-2 border-l-4 border-parkrun-bright pl-4 italic text-gray-600"
          >
            {line.slice(2)}
          </blockquote>
        );
      }
      if (/^\d+\.\s/.test(line)) {
        return (
          <li key={i} className="ml-4 list-decimal">
            {renderInline(line.replace(/^\d+\.\s/, ''))}
          </li>
        );
      }
      if (line.startsWith('- ')) {
        return (
          <li key={i} className="ml-4 list-disc">
            {renderInline(line.slice(2))}
          </li>
        );
      }
      if (line.startsWith('## ')) {
        return (
          <h3 key={i} className="mt-3 font-semibold text-gray-800">
            {line.slice(3)}
          </h3>
        );
      }
      if (line === '') return <br key={i} />;
      return <p key={i}>{renderInline(line)}</p>;
    });
}

function renderInline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) =>
    part.startsWith('**') && part.endsWith('**') ? (
      <strong key={i} className="font-semibold text-gray-900">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <span key={i}>{part}</span>
    ),
  );
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-parkrun-dark px-4 py-3 text-sm text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      {/* Avatar */}
      <div className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-parkrun-bright">
        <span className="text-xs font-bold text-white">p</span>
      </div>

      <div className="flex-1">
        {/* Message bubble */}
        <div className="rounded-2xl rounded-tl-sm bg-white px-4 py-3 text-sm text-gray-700 shadow-sm ring-1 ring-gray-100">
          {message.isLoading ? (
            <div className="flex items-center gap-2 text-gray-400">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-xs">
                {/* Vary the label based on nothing real — just visual variety */}
                Analysing survey data…
              </span>
            </div>
          ) : (
            <div className="prose prose-sm max-w-none leading-relaxed">
              {renderMarkdown(message.content)}
            </div>
          )}
        </div>

        {/* Sources */}
        {!message.isLoading && message.sources && message.sources.length > 0 && (
          <div className="mt-2">
            <div className="flex items-center gap-1.5 text-xs font-medium text-gray-400">
              <BookOpen size={12} />
              Sources
            </div>
            <div className="mt-1.5 flex flex-col gap-1.5">
              {message.sources.map((source) => (
                <div
                  key={source.id}
                  className="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-xs"
                >
                  <p className="font-medium text-gray-700">{source.name}</p>
                  <p className="mt-0.5 text-gray-400 line-clamp-2">
                    {source.excerpt}
                  </p>
                  <div className="mt-1 flex items-center gap-1">
                    <div className="h-1 flex-1 overflow-hidden rounded-full bg-gray-200">
                      <div
                        className="h-full rounded-full bg-parkrun-bright"
                        style={{
                          width: `${Math.round(source.relevanceScore * 100)}%`,
                        }}
                      />
                    </div>
                    <span className="text-gray-400">
                      {Math.round(source.relevanceScore * 100)}% relevant
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
