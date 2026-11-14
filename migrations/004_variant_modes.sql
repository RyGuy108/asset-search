-- "regenerate" re-renders the prompt with text-to-image; "edit" sends the source image to the
-- Magic Hour image editor so the subject itself (the exact product, character, layout) is kept.
ALTER TABLE variant_jobs ADD COLUMN IF NOT EXISTS mode  text NOT NULL DEFAULT 'regenerate';
ALTER TABLE variant_jobs ADD COLUMN IF NOT EXISTS model text;
