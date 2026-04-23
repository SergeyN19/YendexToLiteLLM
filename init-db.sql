-- Creates the two databases used by litellm and litellm-pgvector services.
-- This script runs automatically on first container startup.

CREATE DATABASE litellm_db;
CREATE DATABASE vector_db;
