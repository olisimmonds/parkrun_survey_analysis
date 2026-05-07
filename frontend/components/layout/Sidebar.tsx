'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Upload, Database, MessageSquare, HelpCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

const navLinks = [
  {
    href: '/upload',
    label: 'Upload',
    icon: Upload,
    description: 'Add new datasets',
  },
  {
    href: '/data',
    label: 'Data',
    icon: Database,
    description: 'Manage your datasets',
  },
  {
    href: '/chat',
    label: 'Insights Chat',
    icon: MessageSquare,
    description: 'Ask questions',
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 flex-shrink-0 border-r border-gray-200 bg-white lg:flex lg:flex-col">
      <nav className="flex flex-1 flex-col gap-1 p-4 pt-6">
        <p className="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-gray-400">
          Navigation
        </p>
        {navLinks.map(({ href, label, icon: Icon, description }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                'flex items-center gap-3 rounded-xl px-3 py-3 text-sm transition-all',
                active
                  ? 'bg-parkrun-dark/5 font-semibold text-parkrun-dark'
                  : 'font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900',
              )}
            >
              <span
                className={cn(
                  'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg',
                  active
                    ? 'bg-parkrun-dark text-white'
                    : 'bg-gray-100 text-gray-500',
                )}
              >
                <Icon size={16} />
              </span>
              <span className="flex flex-col">
                <span>{label}</span>
                <span className="text-xs font-normal text-gray-400">
                  {description}
                </span>
              </span>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-gray-100 p-4">
        <a
          href="https://www.parkrun.com"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-gray-400 hover:text-gray-600"
        >
          <HelpCircle size={14} />
          parkrun.com
        </a>
        <p className="mt-2 px-3 text-xs text-gray-300">
          Built by a parkrun Digital Ambassador volunteer
        </p>
      </div>
    </aside>
  );
}
