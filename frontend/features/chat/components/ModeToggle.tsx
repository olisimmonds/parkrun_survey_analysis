'use client';

import { Zap, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useChatStore } from '@/store/chatStore';

export default function ModeToggle() {
  const { config, setMode } = useChatStore();

  return (
    <div className="flex items-center gap-1 rounded-xl border border-gray-200 bg-white p-1 shadow-sm">
      <button
        onClick={() => setMode('standard')}
        className={cn(
          'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
          config.mode === 'standard'
            ? 'bg-parkrun-dark text-white shadow-sm'
            : 'text-gray-500 hover:text-gray-800',
        )}
      >
        <Zap size={13} />
        Standard
      </button>
      <button
        onClick={() => setMode('deep-research')}
        className={cn(
          'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
          config.mode === 'deep-research'
            ? 'bg-parkrun-dark text-white shadow-sm'
            : 'text-gray-500 hover:text-gray-800',
        )}
      >
        <Search size={13} />
        Deep Research
      </button>
    </div>
  );
}
