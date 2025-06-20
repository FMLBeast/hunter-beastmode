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
        # Ensure config is a dictionary - THIS IS THE KEY FIX
        if not isinstance(config, dict):
            if hasattr(config, '__dict__'):
                config = vars(config)
            elif hasattr(config, 'to_dict'):
                config = config.to_dict()
            else:
                raise TypeError('Config must be a dict or have __dict__ or to_dict method')
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize connections as None
        self.sqlite_conn = None
        self.postgres_pool = None
        self.neo4j_driver = None
        
        # Use config.get() for safe access
        self.db_type = self.config.get('type', 'sqlite').lower()
        
        # Initialize database synchronously to avoid cursor issues
        if self.db_type == "sqlite":
            self._init_sqlite_sync()
    
    def _init_sqlite_sync(self):
        """Initialize SQLite database synchronously"""
        try:
            import sqlite3
            
            # Create database directory
            db_path = Path(self.config.get('path', 'data/steganalyzer.db'))
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
        cursor = self.sqlite_conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                target_path TEXT NOT NULL,
                target_dir TEXT,
                status TEXT DEFAULT 'running',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                config TEXT,
                error_message TEXT
            )
        """)
        
        # Files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT,
                file_size INTEGER,
                file_type TEXT,
                mime_type TEXT,
                metadata TEXT,
                analysis_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        # Findings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                method TEXT NOT NULL,
                finding_type TEXT,
                confidence REAL,
                description TEXT,
                evidence TEXT,
                extracted_content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id),
                FOREIGN KEY (file_id) REFERENCES files (id)
            )
        """)
        
        # Checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                checkpoint_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        # Analysis summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_summary (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                summary_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_session ON files (session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_findings_session ON findings (session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_findings_file ON findings (file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_findings_tool ON findings (tool_name)")
        
        self.sqlite_conn.commit()
    
    async def create_session(self, target_path: str, target_dir: str = None, config: dict = None) -> str:
        """Create a new analysis session"""
        session_id = str(uuid.uuid4())
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, target_path, target_dir, config, status)
                VALUES (?, ?, ?, ?, 'running')
            """, (session_id, target_path, target_dir, json.dumps(config or {})))
            self.sqlite_conn.commit()
        
        self.logger.info(f"Created session {session_id} for {target_path}")
        return session_id
    
    async def add_file(self, session_id: str, file_path: str, file_info: dict = None) -> str:
        """Add a file to the session"""
        file_id = str(uuid.uuid4())
        file_info = file_info or {}
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO files (id, session_id, file_path, file_hash, file_size, file_type, mime_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_id, session_id, file_path,
                file_info.get('hash'),
                file_info.get('size'),
                file_info.get('type'),
                file_info.get('mime_type'),
                json.dumps(file_info.get('metadata', {}))
            ))
            self.sqlite_conn.commit()
        
        return file_id
    
    async def store_finding(self, session_id: str, file_id: str, finding: dict):
        """Store an analysis finding"""
        finding_id = str(uuid.uuid4())
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO findings (id, session_id, file_id, tool_name, method, finding_type, 
                                    confidence, description, evidence, extracted_content, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                finding_id, session_id, file_id,
                finding.get('tool_name'),
                finding.get('method'),
                finding.get('type'),
                finding.get('confidence'),
                finding.get('description'),
                finding.get('evidence'),
                finding.get('extracted_content'),
                json.dumps(finding.get('metadata', {}))
            ))
            self.sqlite_conn.commit()
        
        return finding_id
    
    async def store_file_analysis(self, session_id: str, file_info: dict):
        """Store or update file analysis results - THIS IS THE MISSING METHOD"""
        file_path = file_info.get('file_path')
        if not file_path:
            self.logger.error("store_file_analysis: file_info must include 'file_path'.")
            return None
        
        file_id = file_info.get('id')
        
        # Try to find file_id if not provided
        if not file_id:
            if self.db_type == "sqlite":
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT id FROM files WHERE session_id = ? AND file_path = ?", (session_id, file_path))
                row = cursor.fetchone()
                file_id = row[0] if row else None
        
        # If file does not exist, add it
        if not file_id:
            file_id = await self.add_file(session_id, file_path, file_info)
        
        # Update file info
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                UPDATE files SET file_hash = ?, file_size = ?, file_type = ?, mime_type = ?, metadata = ?
                WHERE id = ?
            """, (
                file_info.get('hash'),
                file_info.get('size'),
                file_info.get('type'),
                file_info.get('mime_type'),
                json.dumps(file_info.get('metadata', {})),
                file_id
            ))
            self.sqlite_conn.commit()
        
        self.logger.info(f"store_file_analysis: Updated file analysis for file_id={file_id}, session_id={session_id}")
        return file_id
    
    async def get_completed_methods(self, file_path_or_id: str, session_id: str = None) -> list:
        """Return a list of completed analysis methods for a given file"""
        completed_methods = set()
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            if session_id:
                cursor.execute("SELECT id FROM files WHERE session_id = ? AND file_path = ?", (session_id, file_path_or_id))
                row = cursor.fetchone()
                file_id = row[0] if row else file_path_or_id
            else:
                file_id = file_path_or_id
            cursor.execute("SELECT method FROM findings WHERE file_id = ?", (file_id,))
            completed_methods = {row[0] for row in cursor.fetchall()}
        
        return list(completed_methods)
    
    async def get_session_results(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all results for a session"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT f.*, finding.*
                FROM findings finding
                JOIN files f ON finding.file_id = f.id
                WHERE finding.session_id = ?
                ORDER BY finding.created_at DESC
            """, (session_id,))
            
            results = []
            for row in cursor.fetchall():
                finding = dict(row)
                if finding.get('metadata'):
                    finding['metadata'] = json.loads(finding['metadata'])
                results.append(finding)
            return results
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                session = dict(row)
                if session.get('config'):
                    session['config'] = json.loads(session['config'])
                return session
            return None
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                if session.get('config'):
                    session['config'] = json.loads(session['config'])
                sessions.append(session)
            return sessions
    
    async def store_checkpoint(self, session_id: str, checkpoint_data: dict):
        """Store analysis checkpoint"""
        checkpoint_id = str(uuid.uuid4())
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO checkpoints (id, session_id, checkpoint_data)
                VALUES (?, ?, ?)
            """, (checkpoint_id, session_id, json.dumps(checkpoint_data)))
            self.sqlite_conn.commit()
        
        return checkpoint_id
    
    async def get_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a session"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT * FROM checkpoints 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (session_id,))
            row = cursor.fetchone()
            if row:
                checkpoint = dict(row)
                checkpoint['checkpoint_data'] = json.loads(checkpoint['checkpoint_data'])
                return checkpoint
            return None
    
    async def store_analysis_summary(self, session_id: str, summary: dict):
        """Store analysis summary"""
        summary_id = str(uuid.uuid4())
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_summary (id, session_id, summary_data)
                VALUES (?, ?, ?)
            """, (summary_id, session_id, json.dumps(summary)))
            self.sqlite_conn.commit()
        
        return summary_id
    
    async def close(self):
        """Close database connections"""
        try:
            if self.sqlite_conn:
                self.sqlite_conn.close()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.sqlite_conn:
            try:
                self.sqlite_conn.close()
            except:
                pass