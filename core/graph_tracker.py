"""
Graph Tracker - Tracks relationships between files, findings, and analysis tasks
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
import hashlib

class NodeType(Enum):
    FILE = "file"
    FINDING = "finding"
    TASK = "task"
    TOOL = "tool"
    METHOD = "method"
    SESSION = "session"

class RelationType(Enum):
    CONTAINS = "contains"
    PRODUCES = "produces"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    VALIDATES = "validates"
    CONTRADICTS = "contradicts"
    REFERENCES = "references"

@dataclass
class GraphNode:
    id: str
    type: NodeType
    data: Dict[str, Any]
    created_at: float
    session_id: str

@dataclass
class GraphEdge:
    source: str
    target: str
    relation: RelationType
    weight: float
    metadata: Dict[str, Any]
    created_at: float

class GraphTracker:
    def __init__(self, database):
        self.db = database
        self.logger = logging.getLogger(__name__)
        
        # In-memory graph for fast queries
        self.graph = nx.MultiDiGraph()
        
        # Session-specific subgraphs
        self.session_graphs = {}
        
        # Caching for performance
        self.node_cache = {}
        self.edge_cache = {}
        
        # Configuration
        self.max_cache_size = 10000
        self.similarity_threshold = 0.7
        self.auto_correlation = True
    
    async def create_session_node(self, session_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a session node"""
        node_id = f"session_{session_id}"
        
        node = GraphNode(
            id=node_id,
            type=NodeType.SESSION,
            data=metadata or {},
            created_at=time.time(),
            session_id=session_id
        )
        
        await self._add_node(node)
        
        # Initialize session subgraph
        self.session_graphs[session_id] = nx.MultiDiGraph()
        
        return node_id
    
    async def create_file_node(self, session_id: str, file_path: str, file_info: Dict[str, Any] = None) -> str:
        """Create a file node"""
        # Generate deterministic node ID based on file path and session
        file_hash = hashlib.sha256(f"{session_id}_{file_path}".encode()).hexdigest()[:16]
        node_id = f"file_{file_hash}"
        
        node_data = {
            "file_path": file_path,
            "file_info": file_info or {}
        }
        
        node = GraphNode(
            id=node_id,
            type=NodeType.FILE,
            data=node_data,
            created_at=time.time(),
            session_id=session_id
        )
        
        await self._add_node(node)
        
        # Link to session
        session_node_id = f"session_{session_id}"
        await self._add_edge(session_node_id, node_id, RelationType.CONTAINS, 1.0, {})
        
        return node_id
    
    async def create_task_node(self, session_id: str, file_node_id: str, method: str, tool_name: str, 
                             task_metadata: Dict[str, Any] = None) -> str:
        """Create a task node"""
        task_hash = hashlib.sha256(f"{file_node_id}_{method}_{tool_name}".encode()).hexdigest()[:16]
        node_id = f"task_{task_hash}"
        
        node_data = {
            "method": method,
            "tool_name": tool_name,
            "file_node_id": file_node_id,
            "metadata": task_metadata or {}
        }
        
        node = GraphNode(
            id=node_id,
            type=NodeType.TASK,
            data=node_data,
            created_at=time.time(),
            session_id=session_id
        )
        
        await self._add_node(node)
        
        # Link task to file
        await self._add_edge(file_node_id, node_id, RelationType.PRODUCES, 1.0, {"type": "task_execution"})
        
        return node_id
    
    async def create_finding_node(self, session_id: str, task_node_id: str, finding: Dict[str, Any]) -> str:
        """Create a finding node"""
        finding_hash = hashlib.sha256(
            f"{task_node_id}_{finding.get('type', 'unknown')}_{finding.get('method', 'unknown')}".encode()
        ).hexdigest()[:16]
        node_id = f"finding_{finding_hash}"
        
        node_data = {
            "finding": finding,
            "task_node_id": task_node_id,
            "confidence": finding.get("confidence", 0.0),
            "type": finding.get("type", "unknown"),
            "method": finding.get("method", "unknown"),
            "tool_name": finding.get("tool_name", "unknown")
        }
        
        node = GraphNode(
            id=node_id,
            type=NodeType.FINDING,
            data=node_data,
            created_at=time.time(),
            session_id=session_id
        )
        
        await self._add_node(node)
        
        # Link finding to task
        await self._add_edge(task_node_id, node_id, RelationType.PRODUCES, 
                           finding.get("confidence", 0.5), {"type": "finding_generation"})
        
        # Auto-correlate with existing findings
        if self.auto_correlation:
            await self._correlate_finding(node_id, session_id)
        
        return node_id
    
    async def create_tool_node(self, tool_name: str, tool_metadata: Dict[str, Any] = None) -> str:
        """Create a tool node"""
        node_id = f"tool_{tool_name}"
        
        # Check if tool node already exists
        if node_id in self.node_cache:
            return node_id
        
        node_data = {
            "tool_name": tool_name,
            "metadata": tool_metadata or {}
        }
        
        node = GraphNode(
            id=node_id,
            type=NodeType.TOOL,
            data=node_data,
            created_at=time.time(),
            session_id="global"  # Tools are global
        )
        
        await self._add_node(node)
        return node_id
    
    async def add_dependency_edge(self, source_task_id: str, target_task_id: str, 
                                dependency_type: str = "prerequisite") -> bool:
        """Add dependency relationship between tasks"""
        try:
            await self._add_edge(source_task_id, target_task_id, RelationType.DEPENDS_ON, 1.0, 
                               {"dependency_type": dependency_type})
            return True
        except Exception as e:
            self.logger.error(f"Failed to add dependency edge: {e}")
            return False
    
    async def add_similarity_edge(self, finding_id1: str, finding_id2: str, 
                                similarity_score: float, similarity_reason: str) -> bool:
        """Add similarity relationship between findings"""
        try:
            if similarity_score >= self.similarity_threshold:
                await self._add_edge(finding_id1, finding_id2, RelationType.SIMILAR_TO, 
                                   similarity_score, {"reason": similarity_reason})
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to add similarity edge: {e}")
            return False
    
    async def add_validation_edge(self, finding_id1: str, finding_id2: str, 
                                validation_type: str = "confirms") -> bool:
        """Add validation relationship between findings"""
        try:
            relation = RelationType.VALIDATES if validation_type == "confirms" else RelationType.CONTRADICTS
            await self._add_edge(finding_id1, finding_id2, relation, 0.8, {"type": validation_type})
            return True
        except Exception as e:
            self.logger.error(f"Failed to add validation edge: {e}")
            return False
    
    async def get_file_findings(self, file_node_id: str) -> List[Dict[str, Any]]:
        """Get all findings for a specific file"""
        try:
            findings = []
            
            # Get all tasks for this file
            task_nodes = self._get_connected_nodes(file_node_id, NodeType.TASK, RelationType.PRODUCES)
            
            # Get findings for each task
            for task_id in task_nodes:
                finding_nodes = self._get_connected_nodes(task_id, NodeType.FINDING, RelationType.PRODUCES)
                for finding_id in finding_nodes:
                    if finding_id in self.node_cache:
                        findings.append(self.node_cache[finding_id].data)
            
            return findings
            
        except Exception as e:
            self.logger.error(f"Failed to get file findings: {e}")
            return []
    
    async def get_correlated_findings(self, finding_id: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Get findings correlated with the given finding"""
        try:
            correlated = []
            
            if finding_id not in self.node_cache:
                return []
            
            # Get similar findings
            similar_nodes = self._get_connected_nodes(finding_id, NodeType.FINDING, RelationType.SIMILAR_TO)
            
            # Get validating/contradicting findings
            validating_nodes = self._get_connected_nodes(finding_id, NodeType.FINDING, RelationType.VALIDATES)
            contradicting_nodes = self._get_connected_nodes(finding_id, NodeType.FINDING, RelationType.CONTRADICTS)
            
            all_related = set(similar_nodes + validating_nodes + contradicting_nodes)
            
            for node_id in all_related:
                if node_id in self.node_cache:
                    node_data = self.node_cache[node_id].data
                    if node_data.get("confidence", 0) >= min_confidence:
                        correlated.append({
                            "finding": node_data,
                            "relation_type": self._get_relation_type(finding_id, node_id)
                        })
            
            return correlated
            
        except Exception as e:
            self.logger.error(f"Failed to get correlated findings: {e}")
            return []
    
    async def get_session_analysis_graph(self, session_id: str) -> Dict[str, Any]:
        """Get complete analysis graph for a session"""
        try:
            if session_id not in self.session_graphs:
                return {"nodes": [], "edges": []}
            
            subgraph = self.session_graphs[session_id]
            
            # Convert to serializable format
            nodes = []
            edges = []
            
            for node_id in subgraph.nodes():
                if node_id in self.node_cache:
                    node = self.node_cache[node_id]
                    nodes.append({
                        "id": node.id,
                        "type": node.type.value,
                        "data": node.data,
                        "created_at": node.created_at
                    })
            
            for source, target, key, edge_data in subgraph.edges(keys=True, data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    "relation": edge_data.get("relation", "unknown"),
                    "weight": edge_data.get("weight", 1.0),
                    "metadata": edge_data.get("metadata", {})
                })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "session_id": session_id
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get session analysis graph: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    async def get_analysis_path(self, file_node_id: str, finding_id: str) -> List[Dict[str, Any]]:
        """Get the analysis path from file to specific finding"""
        try:
            if file_node_id not in self.node_cache or finding_id not in self.node_cache:
                return []
            
            # Find path using NetworkX
            try:
                path = nx.shortest_path(self.graph, file_node_id, finding_id)
            except nx.NetworkXNoPath:
                return []
            
            # Build path with node details
            analysis_path = []
            for i, node_id in enumerate(path):
                if node_id in self.node_cache:
                    node = self.node_cache[node_id]
                    path_entry = {
                        "step": i,
                        "node_id": node.id,
                        "type": node.type.value,
                        "data": node.data
                    }
                    
                    # Add edge information if not the last node
                    if i < len(path) - 1:
                        next_node_id = path[i + 1]
                        edge_info = self._get_edge_info(node_id, next_node_id)
                        path_entry["edge_to_next"] = edge_info
                    
                    analysis_path.append(path_entry)
            
            return analysis_path
            
        except Exception as e:
            self.logger.error(f"Failed to get analysis path: {e}")
            return []
    
    async def cleanup_session_graph(self, session_id: str):
        """Clean up graph data for a completed session"""
        try:
            # Remove session subgraph
            if session_id in self.session_graphs:
                del self.session_graphs[session_id]
            
            # Remove session nodes from main graph and cache
            nodes_to_remove = []
            for node_id, node in self.node_cache.items():
                if node.session_id == session_id:
                    nodes_to_remove.append(node_id)
            
            for node_id in nodes_to_remove:
                if self.graph.has_node(node_id):
                    self.graph.remove_node(node_id)
                if node_id in self.node_cache:
                    del self.node_cache[node_id]
            
            # Remove session edges from edge cache
            edges_to_remove = []
            for edge_key in self.edge_cache:
                source, target = edge_key.split("->") if "->" in edge_key else ("", "")
                if (source in nodes_to_remove or target in nodes_to_remove):
                    edges_to_remove.append(edge_key)
            
            for edge_key in edges_to_remove:
                if edge_key in self.edge_cache:
                    del self.edge_cache[edge_key]
            
            self.logger.info(f"Cleaned up graph data for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup session graph: {e}")
    
    async def _add_node(self, node: GraphNode):
        """Add node to graph and cache"""
        self.graph.add_node(node.id, **asdict(node))
        self.node_cache[node.id] = node
        
        # Add to session subgraph
        if node.session_id in self.session_graphs:
            self.session_graphs[node.session_id].add_node(node.id, **asdict(node))
        
        # Manage cache size
        if len(self.node_cache) > self.max_cache_size:
            await self._cleanup_cache()
    
    async def _add_edge(self, source: str, target: str, relation: RelationType, 
                       weight: float, metadata: Dict[str, Any]):
        """Add edge to graph and cache"""
        edge_data = {
            "relation": relation.value,
            "weight": weight,
            "metadata": metadata,
            "created_at": time.time()
        }
        
        self.graph.add_edge(source, target, **edge_data)
        
        # Add to session subgraph if both nodes are in same session
        source_session = self.node_cache.get(source, {}).session_id
        target_session = self.node_cache.get(target, {}).session_id
        
        if source_session == target_session and source_session in self.session_graphs:
            self.session_graphs[source_session].add_edge(source, target, **edge_data)
        
        # Cache edge
        edge_key = f"{source}->{target}"
        self.edge_cache[edge_key] = edge_data
    
    async def _correlate_finding(self, finding_id: str, session_id: str):
        """Auto-correlate finding with existing findings"""
        try:
            if finding_id not in self.node_cache:
                return
            
            current_finding = self.node_cache[finding_id]
            
            # Get all findings in the session
            session_findings = [
                node for node in self.node_cache.values() 
                if node.type == NodeType.FINDING and node.session_id == session_id and node.id != finding_id
            ]
            
            for other_finding in session_findings:
                similarity = self._calculate_finding_similarity(current_finding.data, other_finding.data)
                
                if similarity >= self.similarity_threshold:
                    await self.add_similarity_edge(finding_id, other_finding.id, similarity, "auto_correlation")
                
                # Check for validation/contradiction patterns
                validation_result = self._check_finding_validation(current_finding.data, other_finding.data)
                if validation_result:
                    await self.add_validation_edge(finding_id, other_finding.id, validation_result)
            
        except Exception as e:
            self.logger.debug(f"Failed to correlate finding: {e}")
    
    def _calculate_finding_similarity(self, finding1: Dict[str, Any], finding2: Dict[str, Any]) -> float:
        """Calculate similarity score between two findings"""
        try:
            similarity = 0.0
            
            # Type similarity
            if finding1.get("type") == finding2.get("type"):
                similarity += 0.3
            
            # Method similarity
            if finding1.get("method") == finding2.get("method"):
                similarity += 0.2
            
            # Tool similarity
            if finding1.get("tool_name") == finding2.get("tool_name"):
                similarity += 0.1
            
            # Confidence similarity (closer confidence = more similar)
            conf1 = finding1.get("confidence", 0.0)
            conf2 = finding2.get("confidence", 0.0)
            conf_diff = abs(conf1 - conf2)
            similarity += (1.0 - conf_diff) * 0.2
            
            # File path similarity
            path1 = finding1.get("file_path", "")
            path2 = finding2.get("file_path", "")
            if path1 == path2:
                similarity += 0.2
            
            return min(similarity, 1.0)
            
        except Exception:
            return 0.0
    
    def _check_finding_validation(self, finding1: Dict[str, Any], finding2: Dict[str, Any]) -> Optional[str]:
        """Check if findings validate or contradict each other"""
        try:
            # Same file, same type, different tools = validation
            if (finding1.get("file_path") == finding2.get("file_path") and
                finding1.get("type") == finding2.get("type") and
                finding1.get("tool_name") != finding2.get("tool_name")):
                return "confirms"
            
            # Same file, contradictory types
            contradictory_pairs = [
                ("no_steganography", "steganography_detected"),
                ("clean_file", "suspicious_content"),
                ("valid_format", "format_anomaly")
            ]
            
            type1 = finding1.get("type", "")
            type2 = finding2.get("type", "")
            
            for t1, t2 in contradictory_pairs:
                if (type1 == t1 and type2 == t2) or (type1 == t2 and type2 == t1):
                    return "contradicts"
            
            return None
            
        except Exception:
            return None
    
    def _get_connected_nodes(self, node_id: str, target_type: NodeType, relation: RelationType) -> List[str]:
        """Get nodes connected by specific relation and type"""
        connected = []
        
        try:
            if not self.graph.has_node(node_id):
                return []
            
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                if edge_data and edge_data.get("relation") == relation.value:
                    if neighbor in self.node_cache:
                        neighbor_node = self.node_cache[neighbor]
                        if neighbor_node.type == target_type:
                            connected.append(neighbor)
            
            return connected
            
        except Exception as e:
            self.logger.debug(f"Failed to get connected nodes: {e}")
            return []
    
    def _get_relation_type(self, source: str, target: str) -> str:
        """Get relation type between two nodes"""
        try:
            if self.graph.has_edge(source, target):
                edge_data = self.graph.get_edge_data(source, target)
                return edge_data.get("relation", "unknown")
            return "none"
        except Exception:
            return "unknown"
    
    def _get_edge_info(self, source: str, target: str) -> Dict[str, Any]:
        """Get edge information between two nodes"""
        try:
            if self.graph.has_edge(source, target):
                return dict(self.graph.get_edge_data(source, target))
            return {}
        except Exception:
            return {}
    
    async def _cleanup_cache(self):
        """Clean up cache when it gets too large"""
        try:
            # Remove oldest 20% of cached nodes
            nodes_by_time = sorted(
                self.node_cache.items(),
                key=lambda x: x[1].created_at
            )
            
            remove_count = len(nodes_by_time) // 5
            for node_id, _ in nodes_by_time[:remove_count]:
                if node_id in self.node_cache:
                    del self.node_cache[node_id]
            
            self.logger.debug(f"Cleaned up {remove_count} nodes from cache")
            
        except Exception as e:
            self.logger.debug(f"Failed to cleanup cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        try:
            return {
                "total_nodes": len(self.node_cache),
                "total_edges": self.graph.number_of_edges(),
                "active_sessions": len(self.session_graphs),
                "cache_size": len(self.node_cache),
                "node_types": {
                    node_type.value: sum(1 for node in self.node_cache.values() if node.type == node_type)
                    for node_type in NodeType
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get graph stats: {e}")
            return {"error": str(e)}
