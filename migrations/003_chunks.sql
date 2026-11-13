-- Long prompts (book plates, storyboards) get one extra embedding per sentence, so a
-- query that describes one detail of a 120-word prompt can still match it. The whole
-- prompt's embedding stays on assets.embedding.
CREATE TABLE IF NOT EXISTS asset_chunks (
    asset_id  bigint NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id   text   NOT NULL,
    n         int    NOT NULL,
    text      text   NOT NULL,
    embedding halfvec(256) NOT NULL,
    PRIMARY KEY (asset_id, n)
);
CREATE INDEX IF NOT EXISTS asset_chunks_user ON asset_chunks (user_id);
