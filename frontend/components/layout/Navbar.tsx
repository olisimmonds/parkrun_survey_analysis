'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Upload, Database, MessageSquare, Menu, X } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';

const navLinks = [
  { href: '/upload', label: 'Upload', icon: Upload },
  { href: '/data', label: 'Data', icon: Database },
  { href: '/chat', label: 'Insights Chat', icon: MessageSquare },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-parkrun-dark shadow-md">
      <div className="mx-auto flex h-16 max-w-screen-xl items-center justify-between px-4 sm:px-6">
        {/* Brand */}
        <Link href="/upload" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-parkrun-bright">
            <span className="text-sm font-bold text-white">p</span>
          </div>
          <span className="text-lg font-semibold tracking-tight text-white">
            parkrun <span className="font-light opacity-80">Insights</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-1 md:flex">
          {navLinks.map(({ href, label, icon: Icon }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors',
                  active
                    ? 'bg-white/15 text-white'
                    : 'text-white/70 hover:bg-white/10 hover:text-white',
                )}
              >
                <Icon size={16} />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Mobile menu toggle */}
        <button
          className="flex items-center justify-center rounded-lg p-2 text-white/80 hover:bg-white/10 md:hidden"
          onClick={() => setMobileOpen((o) => !o)}
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile nav */}
      {mobileOpen && (
        <nav className="border-t border-white/10 bg-parkrun-dark md:hidden">
          {navLinks.map(({ href, label, icon: Icon }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                onClick={() => setMobileOpen(false)}
                className={cn(
                  'flex items-center gap-3 px-6 py-3 text-sm font-medium transition-colors',
                  active
                    ? 'bg-white/15 text-white'
                    : 'text-white/70 hover:bg-white/10 hover:text-white',
                )}
              >
                <Icon size={16} />
                {label}
              </Link>
            );
          })}
        </nav>
      )}
    </header>
  );
}
