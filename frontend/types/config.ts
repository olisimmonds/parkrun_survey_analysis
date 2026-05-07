export interface FeatureFlags {
  deepResearch: boolean;
  multiDatasetChat: boolean;
  exportResults: boolean;
  dataVisualisations: boolean;
}

export interface AppConfig {
  apiBaseUrl: string;
  features: FeatureFlags;
  mockMode: boolean;
}

export const defaultConfig: AppConfig = {
  apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL ?? '/api',
  mockMode: process.env.NEXT_PUBLIC_MOCK_MODE !== 'false',
  features: {
    deepResearch: true,
    multiDatasetChat: false,
    exportResults: false,
    dataVisualisations: false,
  },
};
