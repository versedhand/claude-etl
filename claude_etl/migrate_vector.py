#!/usr/bin/env python3
"""Migrate conversation_topics.embedding from 1536 to 3072 dims."""

import psycopg2

conn = psycopg2.connect(
    host='100.127.104.75',
    port=5432,
    dbname='lifedb',
    user='postgres',
    password='StrongPassword123'
)
conn.autocommit = True
cur = conn.cursor()

print("Dropping old column...")
cur.execute('ALTER TABLE conversation_topics DROP COLUMN IF EXISTS embedding')
print("Adding new column with 3072 dims...")
cur.execute('ALTER TABLE conversation_topics ADD COLUMN embedding vector(3072)')
print("Done!")

cur.close()
conn.close()
