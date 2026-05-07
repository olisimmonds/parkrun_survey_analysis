'use client';

import { Send } from 'lucide-react';
import { useState, useRef } from 'react';
import { useChatStore } from '@/store/chatStore';
import { cn } from '@/lib/utils';

const SUGGESTIONS = [
  'What are the main themes in volunteer satisfaction?',
  'How does parkrun participation affect mental wellbeing?',
  'What do new participants say about their first experience?',
  'Which events have the highest volunteer-to-participant ratio?',
];

export default function ChatInput() {
  const { sendMessage, isSending, session } = useChatStore();
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isEmpty = session.messages.length === 0;

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || isSending) return;
    setValue('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    sendMessage(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  };

  return (
    <div className="space-y-3">
      {/* Suggestions (shown when no messages) */}
      {isEmpty && (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => {
                setValue(s);
                textareaRef.current?.focus();
              }}
              className="rounded-xl border border-gray-200 bg-white px-4 py-3 text-left text-xs text-gray-600 shadow-sm hover:border-parkrun-dark/30 hover:bg-gray-50 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input bar */}
      <div className="flex items-end gap-2 rounded-2xl border border-gray-200 bg-white px-4 py-3 shadow-sm focus-within:border-parkrun-dark/40 focus-within:ring-2 focus-within:ring-parkrun-dark/10 transition-all">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={handleTextareaChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your survey data…"
          disabled={isSending}
          className="flex-1 resize-none bg-transparent text-sm text-gray-800 placeholder:text-gray-400 focus:outline-none disabled:opacity-50"
          style={{ minHeight: '24px' }}
        />
        <button
          onClick={handleSubmit}
          disabled={!value.trim() || isSending}
          aria-label="Send"
          className={cn(
            'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg transition-colors',
            value.trim() && !isSending
              ? 'bg-parkrun-dark text-white hover:bg-parkrun-dark/90'
              : 'bg-gray-100 text-gray-300 cursor-not-allowed',
          )}
        >
          <Send size={15} />
        </button>
      </div>
      <p className="text-center text-xs text-gray-300">
        Press Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
