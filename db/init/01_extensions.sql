-- Runs automatically on first container start (via docker-entrypoint-initdb.d)
-- Enables PostGIS extension required for spatial queries.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
