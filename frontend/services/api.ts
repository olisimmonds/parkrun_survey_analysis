import { defaultConfig } from '@/types/config';

export const apiConfig = {
  baseUrl: defaultConfig.apiBaseUrl,
  mockMode: defaultConfig.mockMode,
};

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${apiConfig.baseUrl}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${res.statusText}`);
  // 204 No Content — return undefined cast to T (e.g. for DELETE).
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}
