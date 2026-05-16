-- Remove the restrictive CHECK constraint on surveys.type.
-- The original constraint limited types to a fixed list that doesn't match
-- the values the upload API sends ('survey', 'other', etc.).
-- After this migration, type is free-text.
ALTER TABLE surveys DROP CONSTRAINT IF EXISTS surveys_type_check;
