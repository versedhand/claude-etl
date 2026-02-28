#!/usr/bin/env python3
"""Migrate conversation_topics.embedding from 1536 to 3072 dims."""

import psycopg2

conn = psycopg2.connect(
    host='100.86.218.108',
    port=5432,
    dbname='personal_data',
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
