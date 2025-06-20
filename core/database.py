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
        
        # Fix: Use config.type instead of config.db_type
        self.db_type = config.type.lower()
        
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
    
    async def initialize(self):
        """Initialize database connections asynchronously"""
        try:
            if self.db_type == "sqlite":
                # Already initialized in __init__
                return True
                
            elif self.db_type == "postgresql":
                if not ASYNCPG_AVAILABLE:
                    raise ImportError("asyncpg not available for PostgreSQL support")
                
                await self._init_postgresql()
                
            elif self.db_type == "neo4j":
                if not NEO4J_AVAILABLE:
                    raise ImportError("neo4j not available for Neo4j support")
                
                await self._init_neo4j()
                
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
            self.logger.info(f"Database initialized: {self.db_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False
    
    async def _init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.postgres_pool = await asyncpg.create_pool(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                min_size=2,
                max_size=self.config.connection_pool_size,
                command_timeout=self.config.query_timeout
            )
            
            # Create schema
            async with self.postgres_pool.acquire() as conn:
                await self._create_postgresql_schema(conn)
                
            self.logger.info("PostgreSQL pool initialized")
            
        except Exception as e:
            self.logger.error(f"PostgreSQL initialization failed: {e}")
            raise
    
    async def _create_postgresql_schema(self, conn):
        """Create PostgreSQL database schema"""
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES sessions(id),
            file_id UUID REFERENCES files(id),
            method TEXT NOT NULL,
            finding_type TEXT NOT NULL,
            confidence REAL,
            description TEXT,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS checkpoints (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES sessions(id),
            checkpoint_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
        CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
        CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(finding_type);
        """
        
        await conn.execute(schema)
    
    async def _init_neo4j(self):
        """Initialize Neo4j driver"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=self.config.connection_pool_size
            )
            
            # Test connection
            await self.neo4j_driver.verify_connectivity()
            
            # Create constraints and indexes
            await self._create_neo4j_schema()
            
            self.logger.info("Neo4j driver initialized")
            
        except Exception as e:
            self.logger.error(f"Neo4j initialization failed: {e}")
            raise
    
    async def _create_neo4j_schema(self):
        """Create Neo4j schema constraints and indexes"""
        queries = [
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (fn:Finding) REQUIRE fn.id IS UNIQUE",
            "CREATE INDEX session_status IF NOT EXISTS FOR (s:Session) ON (s.status)",
            "CREATE INDEX finding_type IF NOT EXISTS FOR (fn:Finding) ON (fn.type)",
            "CREATE INDEX file_hash IF NOT EXISTS FOR (f:File) ON (f.hash)"
        ]
        
        async with self.neo4j_driver.session() as session:
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    # Constraints might already exist
                    if "already exists" not in str(e).lower():
                        raise
    
    # Session Management
    async def create_session(self, target_path: str, config: Dict[str, Any], batch_mode: bool = False, target_dir: str = None) -> str:
        """Create a new analysis session"""
        session_id = str(uuid.uuid4())
        
        # Set target_dir if not provided
        if target_dir is None:
            target_dir = str(Path(target_path).parent) if not batch_mode else target_path
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, target_path, config, batch_mode, target_dir)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, target_path, json.dumps(config), batch_mode, target_dir))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sessions (id, target_path, config, batch_mode, target_dir)
                    VALUES ($1, $2, $3, $4, $5)
                """, session_id, target_path, config, batch_mode, target_dir)
                
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                await session.run("""
                    CREATE (s:Session {
                        id: $session_id,
                        target_path: $target_path,
                        config: $config,
                        batch_mode: $batch_mode,
                        target_dir: $target_dir,
                        status: 'running',
                        created_at: datetime()
                    })
                """, session_id=session_id, target_path=target_path, 
                     config=config, batch_mode=batch_mode, target_dir=target_dir)
        
        self.logger.info(f"Created session {session_id} for {target_path}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
                return dict(row) if row else None
                
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (s:Session {id: $session_id})
                    RETURN s
                """, session_id=session_id)
                record = await result.single()
                return dict(record["s"]) if record else None
    
    async def update_session_status(self, session_id: str, status: str, error_message: str = None):
        """Update session status"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            if status == "completed":
                cursor.execute("""
                    UPDATE sessions 
                    SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                    WHERE id = ?
                """, (status, error_message, session_id))
            else:
                cursor.execute("""
                    UPDATE sessions 
                    SET status = ?, error_message = ?
                    WHERE id = ?
                """, (status, error_message, session_id))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                if status == "completed":
                    await conn.execute("""
                        UPDATE sessions 
                        SET status = $1, completed_at = NOW(), error_message = $2
                        WHERE id = $3
                    """, status, error_message, session_id)
                else:
                    await conn.execute("""
                        UPDATE sessions 
                        SET status = $1, error_message = $2
                        WHERE id = $3
                    """, status, error_message, session_id)
                    
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                if status == "completed":
                    await session.run("""
                        MATCH (s:Session {id: $session_id})
                        SET s.status = $status, 
                            s.completed_at = datetime(),
                            s.error_message = $error_message
                    """, session_id=session_id, status=status, error_message=error_message)
                else:
                    await session.run("""
                        MATCH (s:Session {id: $session_id})
                        SET s.status = $status,
                            s.error_message = $error_message
                    """, session_id=session_id, status=status, error_message=error_message)
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM sessions ORDER BY created_at DESC")
                return [dict(row) for row in rows]
                
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (s:Session)
                    RETURN s
                    ORDER BY s.created_at DESC
                """)
                return [dict(record["s"]) async for record in result]
    
    # File Management
    async def add_file(self, session_id: str, file_path: str, file_info: Dict[str, Any]) -> str:
        """Add a file to the session"""
        file_id = str(uuid.uuid4())
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO files (id, session_id, file_path, file_hash, file_size, file_type, mime_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_id, session_id, file_path, 
                  file_info.get('hash'), file_info.get('size'), 
                  file_info.get('type'), file_info.get('mime_type'),
                  json.dumps(file_info.get('metadata', {}))))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO files (id, session_id, file_path, file_hash, file_size, file_type, mime_type, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, file_id, session_id, file_path,
                     file_info.get('hash'), file_info.get('size'),
                     file_info.get('type'), file_info.get('mime_type'),
                     file_info.get('metadata', {}))
                     
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                await session.run("""
                    MATCH (s:Session {id: $session_id})
                    CREATE (f:File {
                        id: $file_id,
                        path: $file_path,
                        hash: $hash,
                        size: $size,
                        type: $type,
                        mime_type: $mime_type,
                        metadata: $metadata,
                        status: 'pending',
                        created_at: datetime()
                    })
                    CREATE (s)-[:CONTAINS]->(f)
                """, session_id=session_id, file_id=file_id, file_path=file_path,
                     hash=file_info.get('hash'), size=file_info.get('size'),
                     type=file_info.get('type'), mime_type=file_info.get('mime_type'),
                     metadata=file_info.get('metadata', {}))
        
        return file_id
    
    async def update_file_status(self, file_id: str, status: str):
        """Update file analysis status"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            if status == "completed":
                cursor.execute("""
                    UPDATE files 
                    SET status = ?, analysis_completed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, file_id))
            elif status == "running":
                cursor.execute("""
                    UPDATE files 
                    SET status = ?, analysis_started = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, file_id))
            else:
                cursor.execute("UPDATE files SET status = ? WHERE id = ?", (status, file_id))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                if status == "completed":
                    await conn.execute("""
                        UPDATE files 
                        SET status = $1, analysis_completed = NOW()
                        WHERE id = $2
                    """, status, file_id)
                elif status == "running":
                    await conn.execute("""
                        UPDATE files 
                        SET status = $1, analysis_started = NOW()
                        WHERE id = $2
                    """, status, file_id)
                else:
                    await conn.execute("UPDATE files SET status = $1 WHERE id = $2", status, file_id)
                    
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                if status == "completed":
                    await session.run("""
                        MATCH (f:File {id: $file_id})
                        SET f.status = $status,
                            f.analysis_completed = datetime()
                    """, file_id=file_id, status=status)
                elif status == "running":
                    await session.run("""
                        MATCH (f:File {id: $file_id})
                        SET f.status = $status,
                            f.analysis_started = datetime()
                    """, file_id=file_id, status=status)
                else:
                    await session.run("""
                        MATCH (f:File {id: $file_id})
                        SET f.status = $status
                    """, file_id=file_id, status=status)
    
    # Finding Management
    async def add_finding(self, session_id: str, file_id: str, finding: Dict[str, Any]) -> str:
        """Add a finding to the database"""
        finding_id = str(uuid.uuid4())
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO findings (id, session_id, file_id, method, finding_type, confidence, description, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (finding_id, session_id, file_id, 
                  finding.get('method'), finding.get('type'),
                  finding.get('confidence'), finding.get('description'),
                  json.dumps(finding.get('metadata', {}))))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO findings (id, session_id, file_id, method, finding_type, confidence, description, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, finding_id, session_id, file_id,
                     finding.get('method'), finding.get('type'),
                     finding.get('confidence'), finding.get('description'),
                     finding.get('metadata', {}))
                     
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                await session.run("""
                    MATCH (s:Session {id: $session_id})
                    MATCH (f:File {id: $file_id})
                    CREATE (fn:Finding {
                        id: $finding_id,
                        method: $method,
                        type: $type,
                        confidence: $confidence,
                        description: $description,
                        metadata: $metadata,
                        created_at: datetime()
                    })
                    CREATE (s)-[:HAS_FINDING]->(fn)
                    CREATE (f)-[:CONTAINS_FINDING]->(fn)
                """, session_id=session_id, file_id=file_id, finding_id=finding_id,
                     method=finding.get('method'), type=finding.get('type'),
                     confidence=finding.get('confidence'), description=finding.get('description'),
                     metadata=finding.get('metadata', {}))
        
        return finding_id
    
    async def get_session_results(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all findings for a session"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT f.*, fi.method, fi.finding_type, fi.confidence, fi.description, fi.metadata as finding_metadata
                FROM files f
                LEFT JOIN findings fi ON f.id = fi.file_id
                WHERE f.session_id = ?
                ORDER BY f.file_path, fi.confidence DESC
            """, (session_id,))
            return [dict(row) for row in cursor.fetchall()]
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT f.*, fi.method, fi.finding_type, fi.confidence, fi.description, fi.metadata as finding_metadata
                    FROM files f
                    LEFT JOIN findings fi ON f.id = fi.file_id
                    WHERE f.session_id = $1
                    ORDER BY f.file_path, fi.confidence DESC
                """, session_id)
                return [dict(row) for row in rows]
                
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (s:Session {id: $session_id})-[:CONTAINS]->(f:File)
                    OPTIONAL MATCH (f)-[:CONTAINS_FINDING]->(fn:Finding)
                    RETURN f, collect(fn) as findings
                    ORDER BY f.path
                """, session_id=session_id)
                
                results = []
                async for record in result:
                    file_data = dict(record["f"])
                    for finding in record["findings"]:
                        if finding:
                            result_item = {**file_data, **dict(finding)}
                            results.append(result_item)
                        else:
                            results.append(file_data)
                return results
    
    async def get_incomplete_files(self, session_id: str) -> List[Dict[str, Any]]:
        """Get files that haven't been fully processed"""
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT * FROM files 
                WHERE session_id = ? AND status != 'completed'
                ORDER BY file_path
            """, (session_id,))
            return [dict(row) for row in cursor.fetchall()]
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM files 
                    WHERE session_id = $1 AND status != 'completed'
                    ORDER BY file_path
                """, session_id)
                return [dict(row) for row in rows]
                
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (s:Session {id: $session_id})-[:CONTAINS]->(f:File)
                    WHERE f.status <> 'completed'
                    RETURN f
                    ORDER BY f.path
                """, session_id=session_id)
                return [dict(record["f"]) async for record in result]
    
    # Checkpoint Management
    async def save_checkpoint(self, session_id: str, checkpoint_data: Dict[str, Any]) -> str:
        """Save analysis checkpoint"""
        checkpoint_id = str(uuid.uuid4())
        
        if self.db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO checkpoints (id, session_id, checkpoint_data)
                VALUES (?, ?, ?)
            """, (checkpoint_id, session_id, json.dumps(checkpoint_data)))
            self.sqlite_conn.commit()
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO checkpoints (id, session_id, checkpoint_data)
                    VALUES ($1, $2, $3)
                """, checkpoint_id, session_id, checkpoint_data)
                
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                await session.run("""
                    MATCH (s:Session {id: $session_id})
                    CREATE (c:Checkpoint {
                        id: $checkpoint_id,
                        data: $checkpoint_data,
                        created_at: datetime()
                    })
                    CREATE (s)-[:HAS_CHECKPOINT]->(c)
                """, session_id=session_id, checkpoint_id=checkpoint_id,
                     checkpoint_data=checkpoint_data)
        
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
            
        elif self.db_type == "postgresql":
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM checkpoints 
                    WHERE session_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, session_id)
                return dict(row) if row else None
                
        elif self.db_type == "neo4j":
            async with self.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (s:Session {id: $session_id})-[:HAS_CHECKPOINT]->(c:Checkpoint)
                    RETURN c
                    ORDER BY c.created_at DESC
                    LIMIT 1
                """, session_id=session_id)
                record = await result.single()
                return dict(record["c"]) if record else None
    
    async def close(self):
        """Close database connections"""
        try:
            if self.sqlite_conn:
                self.sqlite_conn.close()
                
            if self.postgres_pool:
                await self.postgres_pool.close()
                
            if self.neo4j_driver:
                await self.neo4j_driver.close()
                
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