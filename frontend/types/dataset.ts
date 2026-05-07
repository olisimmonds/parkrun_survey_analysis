export type DatasetType = 'Survey' | 'Operational' | 'Document';

export type DatasetStatus = 'ready' | 'processing' | 'error';

export interface Dataset {
  id: string;
  name: string;
  type: DatasetType;
  uploadedAt: string;
  size: number;
  rowCount?: number;
  columnCount?: number;
  description?: string;
  tags?: string[];
  status: DatasetStatus;
  fileName: string;
}

export interface DatasetFilter {
  search: string;
  type: DatasetType | 'All';
}
