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
        self.db_type = config.type
        
        # Connection pools
        self.sqlite_conn = None
        self.postgres_pool = None
        self.neo4j_driver = None
        
        # Initialize based on config
        asyncio.create_task(self._initialize_databases())
    
    async def _initialize_databases(self):
        """Initialize database connections"""
        if self.db_type == "sqlite":
            await self._init_sqlite()
        elif self.db_type == "postgresql" and ASYNCPG_AVAILABLE:
            await self._init_postgresql()
        elif self.db_type == "neo4j" and NEO4J_AVAILABLE:
            await self._init_neo4j()
        else:
            # Fallback to SQLite
            self.logger.warning(f"Database type {self.db_type} not available, falling back to SQLite")
            self.db_type = "sqlite"
            await self._init_sqlite()
        
        # Always initialize Neo4j for graph operations if available
        if NEO4J_AVAILABLE and self.db_type != "neo4j":
            try:
                await self._init_neo4j()
                self.logger.info("Neo4j initialized for graph operations")
            except Exception as e:
                self.logger.warning(f"Could not initialize Neo4j: {e}")
    
    async def _init_sqlite(self):
        """Initialize SQLite database"""
        db_path = Path(self.config.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row
        
        await self._create_sqlite_schema()
        self.logger.info(f"SQLite database initialized at {db_path}")
    
    async def _init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.postgres_pool = await asyncpg.create_pool(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                min_size=1,
                max_size=self.config.connection_pool_size,
                command_timeout=self.config.query_timeout
            )
            
            await self._create_postgresql_schema()
            self.logger.info("PostgreSQL database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _init_neo4j(self):
        """Initialize Neo4j graph database"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            
            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            
            await self._create_neo4j_schema()
            self.logger.info("Neo4j database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j: {e}")
            raise
    
    async def _create_sqlite_schema(self):
        """Create SQLite database schema"""
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            target_path TEXT NOT NULL,
            target_dir TEXT,
            config TEXT,
            status TEXT DEFAULT 'running',
            error_message TEXT,
            batch_mode BOOLEAN DEFAULT FALSE,
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
            tool_name TEXT,
            finding_type TEXT,
            confidence REAL,
            details TEXT,
            raw_output TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id),
            FOREIGN KEY (file_id) REFERENCES files (id)
        );
        
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            method TEXT NOT NULL,
            file_path TEXT,
            status TEXT DEFAULT 'pending',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            result_count INTEGER DEFAULT 0,
            execution_time REAL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        );
        
        CREATE TABLE IF NOT EXISTS correlations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            finding_ids TEXT,
            correlation_type TEXT,
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
        CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        """
        
        cursor = self.sqlite_conn.cursor()
        cursor.executescript(schema)
        self.sqlite_conn.commit()
    
    async def _create_postgresql_schema(self):
        """Create PostgreSQL database schema"""
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY,
            target_path TEXT NOT NULL,
            target_dir TEXT,
            config JSONB,
            status TEXT DEFAULT 'running',
            error_message TEXT,
            batch_mode BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            total_files INTEGER DEFAULT 0,
            processed_files INTEGER DEFAULT 0
        );
        
        CREATE TABLE IF NOT EXISTS files (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES sessions(id),
            file_path TEXT NOT NULL,
            file_hash TEXT,
            file_size BIGINT,
            file_type TEXT,
            mime_type TEXT,
            status TEXT DEFAULT 'pending',
            analysis_started TIMESTAMPTZ,
            analysis_completed TIMESTAMPTZ,
            metadata JSONB
        );
        
        CREATE TABLE IF NOT EXISTS findings (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES sessions(id),
            file_id UUID REFERENCES files(id),
            method TEXT NOT NULL,
            tool_name TEXT,
            finding_type TEXT,
            confidence REAL,
            details TEXT,
            raw_output TEXT,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS tasks (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES sessions(id),
            method TEXT NOT NULL,
            file_path TEXT,
            status TEXT DEFAULT 'pending',
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            error_message TEXT,
            result_count INTEGER DEFAULT 0,
            execution_time REAL
        );
        
        CREATE TABLE IF NOT EXISTS correlations (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES sessions(id),
            finding_ids UUID[],
            correlation_type TEXT,
            confidence REAL,
            description TEXT,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS checkpoints (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES sessions(id),
            checkpoint_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
        CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
        CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(finding_type);
        CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_correlations_session ON correlations(session_id);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema)
    
    async def _create_neo4j_schema(self):
        """Create Neo4j graph schema"""
        constraints = [
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (f:Finding) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT tool_name IF NOT EXISTS FOR (t:Tool) REQUIRE t.name IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX session_status IF NOT EXISTS FOR (s:Session) ON (s.status)",
            "CREATE INDEX finding_type IF NOT EXISTS FOR (f:Finding) ON (f.type)",
            "CREATE INDEX finding_confidence IF NOT EXISTS FOR (f:Finding) ON (f.confidence)"
        ]
        
        with self.neo4j_driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    self.logger.debug(f"Constraint already exists: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    self.logger.debug(f"Index already exists: {e}")
    
    # Session Management
    async def create_session(self, target_path: str, target_dir: str = None, 
                           config: Dict = None, batch_mode: bool = False) -> str:
        """Create a new analysis session"""
        session_id = str(uuid.uuid4())
        config_json = json.dumps(config) if config else None
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, target_path, target_dir, config, batch_mode)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, target_path, target_dir, config_json, batch_mode))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sessions (id, target_path, target_dir, config, batch_mode)
                    VALUES ($1, $2, $3, $4, $5)
                """, session_id, target_path, target_dir, config, batch_mode)
        
        # Create session node in Neo4j
        if self.neo4j_driver:
            with self.neo4j_driver.session() as neo_session:
                neo_session.run("""
                    CREATE (s:Session {
                        id: $session_id,
                        target_path: $target_path,
                        target_dir: $target_dir,
                        batch_mode: $batch_mode,
                        created_at: datetime(),
                        status: 'running'
                    })
                """, session_id=session_id, target_path=target_path, 
                    target_dir=target_dir, batch_mode=batch_mode)
        
        self.logger.info(f"Created session {session_id} for {target_path}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                session = dict(row)
                if session['config']:
                    session['config'] = json.loads(session['config'])
                return session
                
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
                if row:
                    return dict(row)
        
        return None
    
    async def update_session_status(self, session_id: str, status: str, error_message: str = None):
        """Update session status"""
        completed_at = datetime.now(timezone.utc) if status in ['completed', 'failed'] else None
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET status = ?, error_message = ?, completed_at = ?
                WHERE id = ?
            """, (status, error_message, completed_at, session_id))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sessions 
                    SET status = $1, error_message = $2, completed_at = $3
                    WHERE id = $4
                """, status, error_message, completed_at, session_id)
        
        # Update Neo4j
        if self.neo4j_driver:
            with self.neo4j_driver.session() as neo_session:
                neo_session.run("""
                    MATCH (s:Session {id: $session_id})
                    SET s.status = $status, s.completed_at = datetime()
                """, session_id=session_id, status=status)
    
    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List analysis sessions"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions 
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                if session['config']:
                    session['config'] = json.loads(session['config'])
                sessions.append(session)
            return sessions
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM sessions 
                    ORDER BY created_at DESC LIMIT $1
                """, limit)
                return [dict(row) for row in rows]
        
        return []
    
    # File Management
    async def store_file_info(self, session_id: str, file_path: str, file_info: Dict[str, Any]) -> str:
        """Store file information"""
        file_id = str(uuid.uuid4())
        
        # Calculate file hash
        file_hash = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
            hash_value = file_hash.hexdigest()
        except Exception:
            hash_value = None
        
        metadata_json = json.dumps(file_info)
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO files (id, session_id, file_path, file_hash, file_size, 
                                 file_type, mime_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_id, session_id, file_path, hash_value, 
                  file_info.get('size'), file_info.get('type'), 
                  file_info.get('mime_type'), metadata_json))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO files (id, session_id, file_path, file_hash, file_size,
                                     file_type, mime_type, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, file_id, session_id, file_path, hash_value,
                    file_info.get('size'), file_info.get('type'),
                    file_info.get('mime_type'), file_info)
        
        # Create file node in Neo4j
        if self.neo4j_driver:
            with self.neo4j_driver.session() as neo_session:
                neo_session.run("""
                    MATCH (s:Session {id: $session_id})
                    CREATE (f:File {
                        id: $file_id,
                        path: $file_path,
                        hash: $hash_value,
                        size: $file_size,
                        type: $file_type,
                        mime_type: $mime_type
                    })
                    CREATE (s)-[:ANALYZES]->(f)
                """, session_id=session_id, file_id=file_id, file_path=file_path,
                    hash_value=hash_value, file_size=file_info.get('size'),
                    file_type=file_info.get('type'), mime_type=file_info.get('mime_type'))
        
        return file_id
    
    # Findings Management
    async def store_findings(self, session_id: str, method: str, findings: List[Dict[str, Any]]):
        """Store analysis findings"""
        if not findings:
            return
        
        # Get file ID if file_path is provided
        file_id = None
        if findings and 'file_path' in findings[0]:
            file_id = await self._get_file_id(session_id, findings[0]['file_path'])
        
        for finding in findings:
            finding_id = str(uuid.uuid4())
            metadata_json = json.dumps(finding.get('metadata', {}))
            
            if self.db_type == "sqlite":
                cursor = self.sqlite_conn.cursor()
                cursor.execute("""
                    INSERT INTO findings (id, session_id, file_id, method, tool_name,
                                        finding_type, confidence, details, raw_output, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (finding_id, session_id, file_id, method,
                      finding.get('tool_name'), finding.get('type'),
                      finding.get('confidence'), finding.get('details'),
                      finding.get('raw_output'), metadata_json))
                self.sqlite_conn.commit()
                
            elif self.db_type == "postgresql":
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO findings (id, session_id, file_id, method, tool_name,
                                            finding_type, confidence, details, raw_output, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, finding_id, session_id, file_id, method,
                        finding.get('tool_name'), finding.get('type'),
                        finding.get('confidence'), finding.get('details'),
                        finding.get('raw_output'), finding.get('metadata', {}))
            
            # Create finding node in Neo4j
            if self.neo4j_driver:
                with self.neo4j_driver.session() as neo_session:
                    if file_id:
                        neo_session.run("""
                            MATCH (f:File {id: $file_id})
                            CREATE (finding:Finding {
                                id: $finding_id,
                                method: $method,
                                type: $finding_type,
                                confidence: $confidence,
                                details: $details,
                                created_at: datetime()
                            })
                            CREATE (f)-[:HAS_FINDING]->(finding)
                        """, file_id=file_id, finding_id=finding_id, method=method,
                            finding_type=finding.get('type'), confidence=finding.get('confidence'),
                            details=finding.get('details'))
                    else:
                        neo_session.run("""
                            MATCH (s:Session {id: $session_id})
                            CREATE (finding:Finding {
                                id: $finding_id,
                                method: $method,
                                type: $finding_type,
                                confidence: $confidence,
                                details: $details,
                                created_at: datetime()
                            })
                            CREATE (s)-[:HAS_FINDING]->(finding)
                        """, session_id=session_id, finding_id=finding_id, method=method,
                            finding_type=finding.get('type'), confidence=finding.get('confidence'),
                            details=finding.get('details'))
    
    async def get_session_results(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all findings for a session"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT f.*, files.file_path 
                FROM findings f
                LEFT JOIN files ON f.file_id = files.id
                WHERE f.session_id = ?
                ORDER BY f.created_at
            """, (session_id,))
            findings = []
            for row in cursor.fetchall():
                finding = dict(row)
                if finding['metadata']:
                    finding['metadata'] = json.loads(finding['metadata'])
                findings.append(finding)
            return findings
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT f.*, files.file_path 
                    FROM findings f
                    LEFT JOIN files ON f.file_id = files.id
                    WHERE f.session_id = $1
                    ORDER BY f.created_at
                """, session_id)
                return [dict(row) for row in rows]
        
        return []
    
    # Task Management
    async def record_task_completion(self, session_id: str, method: str):
        """Record task completion"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                UPDATE tasks SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND method = ?
            """, (session_id, method))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE tasks SET status = 'completed', completed_at = NOW()
                    WHERE session_id = $1 AND method = $2
                """, session_id, method)
    
    async def record_task_failure(self, session_id: str, method: str, error_message: str):
        """Record task failure"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                UPDATE tasks SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
                                error_message = ?
                WHERE session_id = ? AND method = ?
            """, (error_message, session_id, method))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE tasks SET status = 'failed', completed_at = NOW(),
                                    error_message = $1
                    WHERE session_id = $2 AND method = $3
                """, error_message, session_id, method)
    
    async def get_incomplete_files(self, session_id: str) -> List[str]:
        """Get list of incomplete file paths for session"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT DISTINCT file_path FROM files 
                WHERE session_id = ? AND status != 'completed'
            """, (session_id,))
            return [row[0] for row in cursor.fetchall()]
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT DISTINCT file_path FROM files 
                    WHERE session_id = $1 AND status != 'completed'
                """, session_id)
                return [row['file_path'] for row in rows]
        
        return []
    
    # Utility methods
    async def _get_file_id(self, session_id: str, file_path: str) -> Optional[str]:
        """Get file ID by path"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT id FROM files WHERE session_id = ? AND file_path = ?
            """, (session_id, file_path))
            row = cursor.fetchone()
            return row[0] if row else None
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id FROM files WHERE session_id = $1 AND file_path = $2
                """, session_id, file_path)
                return row['id'] if row else None
        
        return None
    
    # Cleanup
    async def close(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        if self.neo4j_driver:
            self.neo4j_driver.close()
        
        self.logger.info("Database connections closed")