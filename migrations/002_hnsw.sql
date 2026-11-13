-- Built after bulk load: building HNSW on a full table is far faster than maintaining it row by row.
CREATE INDEX IF NOT EXISTS assets_embedding_hnsw ON assets USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
