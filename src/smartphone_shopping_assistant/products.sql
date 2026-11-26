CREATE TABLE IF NOT EXISTS public.products (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    ram_gb INTEGER CHECK (ram_gb > 0),
    storage_gb INTEGER CHECK (storage_gb > 0),
    color TEXT NOT NULL,
    carrier_unlocked BOOLEAN NOT NULL,
    price NUMERIC NOT NULL CHECK (price > 0 AND price < 'Infinity'::numeric),
    currency TEXT NOT NULL,
    currency_status TEXT NOT NULL,
    source_currency TEXT,
    conversion_applied BOOLEAN NOT NULL,
    raw_fields JSONB NOT NULL,
    text TEXT NOT NULL
);
