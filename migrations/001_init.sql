CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS assets (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     text        NOT NULL,
    key         text,                       -- caller's own id (unique per user)
    mh_id       text,                       -- Magic Hour project id
    kind        text        NOT NULL CHECK (kind IN ('image', 'video', 'audio')),
    tool        text        NOT NULL,       -- ai_image_generator, face_swap_photo, image_to_video, ...
    model       text,
    prompt      text,
    search_text text        NOT NULL,       -- prompt, or text inherited from parents when there is none
    parent_ids  bigint[]    NOT NULL DEFAULT '{}',
    path        text,
    url         text,
    credits     real,
    created_at  timestamptz NOT NULL DEFAULT now(),
    embedding   halfvec(256),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    UNIQUE (user_id, key)
);

CREATE INDEX IF NOT EXISTS assets_user_created ON assets (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS assets_tsv ON assets USING gin (tsv);
CREATE INDEX IF NOT EXISTS assets_parents ON assets USING gin (parent_ids);

CREATE TABLE IF NOT EXISTS variant_jobs (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     text        NOT NULL,
    source_id   bigint      NOT NULL REFERENCES assets(id),
    instruction text,
    prompts     jsonb       NOT NULL,
    status      text        NOT NULL DEFAULT 'queued',
    mh_ids      text[]      NOT NULL DEFAULT '{}',
    result_ids  bigint[]    NOT NULL DEFAULT '{}',
    error       text,
    credits     real,
    created_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);
