#!/usr/bin/env python3
"""Migrate message_embeddings.embedding from 1536 to 3072 dims."""

import os
import psycopg2

conn = psycopg2.connect(
    host='100.127.104.75',
    port=5432,
    dbname='lifedb',
    user='postgres',
    password=os.environ['LIFEDB_PASSWORD']
)
conn.autocommit = True
cur = conn.cursor()

print("Dropping old embedding column...")
cur.execute('ALTER TABLE message_embeddings DROP COLUMN IF EXISTS embedding')
print("Adding new column with 3072 dims...")
cur.execute('ALTER TABLE message_embeddings ADD COLUMN embedding vector(3072)')
print("Done!")

cur.close()
conn.close()
