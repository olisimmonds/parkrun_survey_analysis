-- Enable pgvector for embedding storage and similarity search.
-- Must run before any table that uses the VECTOR type.
CREATE EXTENSION IF NOT EXISTS vector;
