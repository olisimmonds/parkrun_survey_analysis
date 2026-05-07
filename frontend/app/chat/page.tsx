import ChatWindow from '@/features/chat/components/ChatWindow';

export default function ChatPage() {
  return (
    <div className="flex h-full flex-col">
      {/* Page header */}
      <div className="border-b border-gray-100 bg-white px-6 py-4">
        <h1 className="text-lg font-bold text-gray-900">Insights Chat</h1>
        <p className="text-sm text-gray-400">
          Ask questions about your survey data in natural language
        </p>
      </div>

      {/* Chat window fills remaining height */}
      <div className="flex-1 overflow-hidden">
        <ChatWindow />
      </div>
    </div>
  );
}
