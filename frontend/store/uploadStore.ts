'use client';

import { create } from 'zustand';
import type { UploadedFile } from '@/services/upload.service';
import { uploadService } from '@/services/upload.service';
import { useDatasetStore } from './datasetStore';

interface UploadState {
  uploads: UploadedFile[];
  addFiles: (files: File[]) => void;
  removeUpload: (id: string) => void;
  clearCompleted: () => void;
}

export const useUploadStore = create<UploadState>((set, get) => ({
  uploads: [],

  addFiles: (files) => {
    const newUploads: UploadedFile[] = files.map((file) => ({
      id: Math.random().toString(36).slice(2),
      file,
      progress: 0,
      status: 'pending',
    }));

    set((state) => ({ uploads: [...state.uploads, ...newUploads] }));

    newUploads.forEach((upload) => {
      set((state) => ({
        uploads: state.uploads.map((u) =>
          u.id === upload.id ? { ...u, status: 'uploading' } : u,
        ),
      }));

      uploadService
        .uploadFile(upload.file, (progress) => {
          set((state) => ({
            uploads: state.uploads.map((u) =>
              u.id === upload.id ? { ...u, progress } : u,
            ),
          }));
        })
        .then((dataset) => {
          useDatasetStore.getState().addDataset(dataset);
          set((state) => ({
            uploads: state.uploads.map((u) =>
              u.id === upload.id
                ? { ...u, status: 'complete', progress: 100, dataset }
                : u,
            ),
          }));
        })
        .catch((err) => {
          set((state) => ({
            uploads: state.uploads.map((u) =>
              u.id === upload.id
                ? { ...u, status: 'error', error: String(err) }
                : u,
            ),
          }));
        });
    });
  },

  removeUpload: (id) =>
    set((state) => ({ uploads: state.uploads.filter((u) => u.id !== id) })),

  clearCompleted: () =>
    set((state) => ({
      uploads: state.uploads.filter((u) => u.status !== 'complete'),
    })),
}));
