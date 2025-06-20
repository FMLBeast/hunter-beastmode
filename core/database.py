"""
Database Management System for StegAnalyzer
Supports SQLite, PostgreSQL, and Neo4j for comprehensive data storage and relationship tracking
"""

import asyncio
import sqlite3
import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
import uuid
from dataclasses import asdict

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize connections as None
        self.sqlite_conn = None
        self.postgres_pool = None
        self.neo4j_driver = None
        
        # Determine database type
        self.db_type = config.db_type.lower()
        
        # Initialize database synchronously to avoid cursor issues
        if self.db_type == "sqlite":
            self._init_sqlite_sync()
        # For other types, we'll initialize async later
    
    def _init_sqlite_sync(self):
        """Initialize SQLite database synchronously"""
        try:
            import sqlite3
            
            # Create database directory
            db_path = Path(self.config.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize connection
            self.sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            
            # Create schema
            self._create_sqlite_schema_sync()
            self.logger.info(f"SQLite database initialized at {db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    def _create_sqlite_schema_sync(self):
        """Create SQLite database schema synchronously"""
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            target_path TEXT NOT NULL,
            target_dir TEXT,
            config TEXT,
            status TEXT DEFAULT 'running',
            error_message TEXT,
            batch_mode BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            total_files INTEGER DEFAULT 0,
            processed_files INTEGER DEFAULT 0
        );
        
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT,
            file_size INTEGER,
            file_type TEXT,
            mime_type TEXT,
            status TEXT DEFAULT 'pending',
            analysis_started TIMESTAMP,
            analysis_completed TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        );
        
        CREATE TABLE IF NOT EXISTS findings (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            file_id TEXT,
            method TEXT NOT NULL,
            finding_type TEXT NOT NULL,
            confidence REAL,
            description TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        );
        
        CREATE TABLE IF NOT EXISTS checkpoints (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            checkpoint_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
        CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
        CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(finding_type);
        """
        
        cursor = self.sqlite_conn.cursor()
        cursor.executescript(schema)
        self.sqlite_conn.commit()