'use client';

import { useEffect, useRef } from 'react';
import { useChatStore } from '@/store/chatStore';
import ChatMessageComponent from './ChatMessage';
import ChatInput from './ChatInput';
import ModeToggle from './ModeToggle';
import { RotateCcw, MessageSquare } from 'lucide-react';

export default function ChatWindow() {
  const { session, clearSession, config } = useChatStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [session.messages]);

  return (
    <div className="flex h-full flex-col">
      {/* Chat toolbar */}
      <div className="flex items-center justify-between border-b border-gray-100 bg-white px-6 py-3">
        <div className="flex items-center gap-3">
          <ModeToggle />
          {config.mode === 'deep-research' && (
            <span className="rounded-full bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-700">
              Deeper analysis · takes a little longer
            </span>
          )}
        </div>
        {session.messages.length > 0 && (
          <button
            onClick={clearSession}
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors"
          >
            <RotateCcw size={12} />
            New chat
          </button>
        )}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {session.messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-parkrun-dark/5">
              <MessageSquare size={28} className="text-parkrun-dark/40" />
            </div>
            <div>
              <p className="font-semibold text-gray-700">
                Ask a question about your data
              </p>
              <p className="mt-1 max-w-sm text-sm text-gray-400">
                Upload survey datasets on the Data page, then ask questions here
                to generate AI-powered insights.
              </p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-6">
            {session.messages.map((msg) => (
              <ChatMessageComponent key={msg.id} message={msg} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="border-t border-gray-100 bg-white px-6 py-4">
        <ChatInput />
      </div>
    </div>
  );
}
