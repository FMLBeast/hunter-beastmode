"""
Checkpoint Manager - Session state persistence and recovery
"""

import json
import logging
import pickle
import time
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import gzip
import tempfile

class CheckpointManager:
    def __init__(self, database):
        self.db = database
        self.logger = logging.getLogger(__name__)
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint configuration
        self.max_checkpoints_per_session = 10
        self.checkpoint_interval = 300  # 5 minutes
        self.compression_enabled = True
        
        # In-memory cache for quick access
        self.checkpoint_cache = {}
        self.last_checkpoint_time = {}
    
    async def save_checkpoint(self, session_id: str, session_state: Dict[str, Any]) -> bool:
        """Save session checkpoint"""
        try:
            # Check if enough time has passed since last checkpoint
            current_time = time.time()
            last_time = self.last_checkpoint_time.get(session_id, 0)
            
            if current_time - last_time < self.checkpoint_interval:
                return False  # Skip checkpoint, too soon
            
            checkpoint_id = self._generate_checkpoint_id(session_id, current_time)
            checkpoint_data = self._prepare_checkpoint_data(session_state)
            
            # Save to database
            await self._save_checkpoint_to_db(session_id, checkpoint_id, checkpoint_data)
            
            # Save to filesystem as backup
            await self._save_checkpoint_to_file(session_id, checkpoint_id, checkpoint_data)
            
            # Update cache and timestamp
            self.checkpoint_cache[session_id] = checkpoint_data
            self.last_checkpoint_time[session_id] = current_time
            
            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints(session_id)
            
            self.logger.info(f"Checkpoint saved for session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for session {session_id}: {e}")
            return False
    
    async def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint for session"""
        try:
            # Try cache first
            if session_id in self.checkpoint_cache:
                self.logger.debug(f"Loaded checkpoint from cache for session {session_id}")
                return self.checkpoint_cache[session_id]
            
            # Try database
            checkpoint = await self._load_checkpoint_from_db(session_id)
            if checkpoint:
                self.checkpoint_cache[session_id] = checkpoint
                self.logger.info(f"Loaded checkpoint from database for session {session_id}")
                return checkpoint
            
            # Try filesystem as fallback
            checkpoint = await self._load_checkpoint_from_file(session_id)
            if checkpoint:
                self.checkpoint_cache[session_id] = checkpoint
                self.logger.info(f"Loaded checkpoint from file for session {session_id}")
                return checkpoint
            
            self.logger.warning(f"No checkpoint found for session {session_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for session {session_id}: {e}")
            return None
    
    async def list_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a session"""
        try:
            checkpoints = []
            
            # Get from database
            if hasattr(self.db, 'list_checkpoints'):
                db_checkpoints = await self.db.list_checkpoints(session_id)
                checkpoints.extend(db_checkpoints)
            
            # Get from filesystem
            file_checkpoints = await self._list_file_checkpoints(session_id)
            checkpoints.extend(file_checkpoints)
            
            # Remove duplicates and sort by timestamp
            seen_ids = set()
            unique_checkpoints = []
            
            for checkpoint in checkpoints:
                if checkpoint['id'] not in seen_ids:
                    unique_checkpoints.append(checkpoint)
                    seen_ids.add(checkpoint['id'])
            
            return sorted(unique_checkpoints, key=lambda x: x.get('timestamp', 0), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints for session {session_id}: {e}")
            return []
    
    async def delete_checkpoint(self, session_id: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        try:
            success = True
            
            # Delete from database
            if hasattr(self.db, 'delete_checkpoint'):
                success &= await self.db.delete_checkpoint(session_id, checkpoint_id)
            
            # Delete from filesystem
            success &= await self._delete_file_checkpoint(session_id, checkpoint_id)
            
            # Remove from cache
            if session_id in self.checkpoint_cache:
                del self.checkpoint_cache[session_id]
            
            if success:
                self.logger.info(f"Deleted checkpoint {checkpoint_id} for session {session_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def cleanup_session_checkpoints(self, session_id: str) -> bool:
        """Delete all checkpoints for a session"""
        try:
            checkpoints = await self.list_checkpoints(session_id)
            
            for checkpoint in checkpoints:
                await self.delete_checkpoint(session_id, checkpoint['id'])
            
            # Clear cache
            if session_id in self.checkpoint_cache:
                del self.checkpoint_cache[session_id]
            
            if session_id in self.last_checkpoint_time:
                del self.last_checkpoint_time[session_id]
            
            self.logger.info(f"Cleaned up all checkpoints for session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints for session {session_id}: {e}")
            return False
    
    def _generate_checkpoint_id(self, session_id: str, timestamp: float) -> str:
        """Generate unique checkpoint ID"""
        data = f"{session_id}_{timestamp}".encode()
        return hashlib.sha256(data).hexdigest()[:16]
    
    def _prepare_checkpoint_data(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare session state for storage"""
        checkpoint = {
            "timestamp": time.time(),
            "version": "1.0",
            "session_state": {}
        }
        
        # Filter and serialize session state
        serializable_state = {}
        
        for key, value in session_state.items():
            try:
                # Convert Path objects to strings
                if hasattr(value, '__fspath__'):
                    serializable_state[key] = str(value)
                # Convert sets to lists
                elif isinstance(value, set):
                    serializable_state[key] = list(value)
                # Keep serializable types
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_state[key] = value
                # Skip non-serializable objects
                else:
                    self.logger.debug(f"Skipping non-serializable field: {key}")
                    continue
                    
            except Exception as e:
                self.logger.debug(f"Failed to serialize field {key}: {e}")
                continue
        
        checkpoint["session_state"] = serializable_state
        return checkpoint
    
    async def _save_checkpoint_to_db(self, session_id: str, checkpoint_id: str, checkpoint_data: Dict[str, Any]):
        """Save checkpoint to database"""
        try:
            data_str = json.dumps(checkpoint_data, default=str)
            
            if hasattr(self.db, 'save_checkpoint'):
                await self.db.save_checkpoint(session_id, checkpoint_id, data_str)
            else:
                # Fallback for databases without checkpoint support
                self.logger.debug("Database doesn't support checkpoints, using file storage only")
                
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint to database: {e}")
    
    async def _save_checkpoint_to_file(self, session_id: str, checkpoint_id: str, checkpoint_data: Dict[str, Any]):
        """Save checkpoint to filesystem"""
        try:
            session_dir = self.checkpoint_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            checkpoint_file = session_dir / f"{checkpoint_id}.json"
            
            # Serialize and optionally compress
            data_str = json.dumps(checkpoint_data, indent=2, default=str)
            
            if self.compression_enabled:
                checkpoint_file = checkpoint_file.with_suffix('.json.gz')
                with gzip.open(checkpoint_file, 'wt', encoding='utf-8') as f:
                    f.write(data_str)
            else:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    f.write(data_str)
                    
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint to file: {e}")
    
    async def _load_checkpoint_from_db(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint from database"""
        try:
            if hasattr(self.db, 'load_latest_checkpoint'):
                data_str = await self.db.load_latest_checkpoint(session_id)
                if data_str:
                    return json.loads(data_str)["session_state"]
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to load checkpoint from database: {e}")
            return None
    
    async def _load_checkpoint_from_file(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint from filesystem"""
        try:
            session_dir = self.checkpoint_dir / session_id
            if not session_dir.exists():
                return None
            
            # Find latest checkpoint file
            checkpoint_files = []
            for ext in ['*.json.gz', '*.json']:
                checkpoint_files.extend(session_dir.glob(ext))
            
            if not checkpoint_files:
                return None
            
            # Sort by modification time, newest first
            latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            
            # Load and decompress if needed
            if latest_file.suffix == '.gz':
                with gzip.open(latest_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            return data["session_state"]
            
        except Exception as e:
            self.logger.debug(f"Failed to load checkpoint from file: {e}")
            return None
    
    async def _list_file_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """List checkpoints from filesystem"""
        try:
            session_dir = self.checkpoint_dir / session_id
            if not session_dir.exists():
                return []
            
            checkpoints = []
            for checkpoint_file in session_dir.glob('*.json*'):
                try:
                    stat = checkpoint_file.stat()
                    checkpoint_id = checkpoint_file.stem.replace('.json', '')
                    
                    checkpoints.append({
                        'id': checkpoint_id,
                        'timestamp': stat.st_mtime,
                        'size': stat.st_size,
                        'source': 'file'
                    })
                except Exception as e:
                    self.logger.debug(f"Failed to process checkpoint file {checkpoint_file}: {e}")
                    continue
            
            return checkpoints
            
        except Exception as e:
            self.logger.debug(f"Failed to list file checkpoints: {e}")
            return []
    
    async def _delete_file_checkpoint(self, session_id: str, checkpoint_id: str) -> bool:
        """Delete checkpoint file"""
        try:
            session_dir = self.checkpoint_dir / session_id
            
            for ext in ['.json', '.json.gz']:
                checkpoint_file = session_dir / f"{checkpoint_id}{ext}"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Failed to delete checkpoint file: {e}")
            return False
    
    async def _cleanup_old_checkpoints(self, session_id: str):
        """Remove old checkpoints to maintain limit"""
        try:
            checkpoints = await self.list_checkpoints(session_id)
            
            if len(checkpoints) > self.max_checkpoints_per_session:
                # Sort by timestamp, keep newest
                checkpoints.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
                
                # Delete excess checkpoints
                for checkpoint in checkpoints[self.max_checkpoints_per_session:]:
                    await self.delete_checkpoint(session_id, checkpoint['id'])
                    
        except Exception as e:
            self.logger.debug(f"Failed to cleanup old checkpoints: {e}")
    
    async def force_checkpoint(self, session_id: str, session_state: Dict[str, Any]) -> bool:
        """Force immediate checkpoint regardless of timing"""
        # Temporarily override the interval check
        original_time = self.last_checkpoint_time.get(session_id, 0)
        self.last_checkpoint_time[session_id] = 0
        
        result = await self.save_checkpoint(session_id, session_state)
        
        # Restore original time if checkpoint failed
        if not result:
            self.last_checkpoint_time[session_id] = original_time
        
        return result
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint system statistics"""
        try:
            stats = {
                "total_sessions_in_cache": len(self.checkpoint_cache),
                "checkpoint_directory": str(self.checkpoint_dir),
                "compression_enabled": self.compression_enabled,
                "checkpoint_interval": self.checkpoint_interval,
                "max_checkpoints_per_session": self.max_checkpoints_per_session
            }
            
            # Calculate total checkpoint files
            total_files = 0
            total_size = 0
            
            for session_dir in self.checkpoint_dir.iterdir():
                if session_dir.is_dir():
                    for checkpoint_file in session_dir.glob('*.json*'):
                        total_files += 1
                        total_size += checkpoint_file.stat().st_size
            
            stats.update({
                "total_checkpoint_files": total_files,
                "total_storage_size": total_size,
                "average_file_size": total_size / total_files if total_files > 0 else 0
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get checkpoint stats: {e}")
            return {"error": str(e)}
