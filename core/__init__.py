"""
Core analysis components
"""
# Use the actual renamed files
from .file_analyzer import FileAnalyzer
from .database import DatabaseManager  # This is database.py, not steg_database.py
from .graph_tracker import GraphTracker

# Optional components with dependencies
try:
    from .orchestrator import StegOrchestrator
except ImportError:
    StegOrchestrator = None

try:
    from .dashboard import Dashboard
except ImportError:
    Dashboard = None

try:
    from .reporter import ReportGenerator
except ImportError:
    ReportGenerator = None

__all__ = ['FileAnalyzer', 'DatabaseManager', 'GraphTracker']
if StegOrchestrator:
    __all__.append('StegOrchestrator')
if Dashboard:
    __all__.append('Dashboard')  
if ReportGenerator:
    __all__.append('ReportGenerator')
