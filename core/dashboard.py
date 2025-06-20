"""
Live Dashboard - Real-time Analysis Monitoring and Visualization
FastAPI-based web dashboard with WebSocket updates and interactive visualizations
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import webbrowser
import threading

class Dashboard:
    def __init__(self, config, database):
        self.config = config.dashboard
        self.db = database
        self.logger = logging.getLogger(__name__)
        
        # FastAPI app
        self.app = FastAPI(title="StegAnalyzer Dashboard")
        self.templates = Jinja2Templates(directory="templates")
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Dashboard state
        self.current_session = None
        self.session_stats = {}
        self.real_time_logs = []
        self.max_logs = self.config.max_log_entries
        
        # Server instance
        self.server = None
        self.server_thread = None
        self.url = f"http://{self.config.host}:{self.config.port}"
        
        # Setup routes
        self._setup_routes()
        
        # Create templates directory and files
        self._create_dashboard_files()
    
    def _create_dashboard_files(self):
        """Create dashboard template and static files"""
        # Create directories
        Path("templates").mkdir(exist_ok=True)
        Path("static").mkdir(exist_ok=True)
        Path("static/css").mkdir(exist_ok=True)
        Path("static/js").mkdir(exist_ok=True)
        
        # Create main template
        self._create_main_template()
        self._create_static_files()
    
    def _create_main_template(self):
        """Create the main dashboard HTML template"""
        template_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StegAnalyzer Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link href="/static/css/dashboard.css" rel="stylesheet">
</head>
<body class="bg-gray-900 text-white">
    <!-- Header -->
    <header class="bg-gray-800 shadow-lg border-b border-gray-700">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <i class="fas fa-search text-blue-500 text-2xl mr-3"></i>
                    <h1 class="text-2xl font-bold text-white">StegAnalyzer</h1>
                    <span class="ml-4 px-3 py-1 bg-blue-600 rounded-full text-sm" id="status-badge">
                        <i class="fas fa-circle text-green-400 mr-1"></i>
                        Active
                    </span>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="text-sm text-gray-300" id="session-info">
                        No active session
                    </div>
                    <button id="export-btn" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors">
                        <i class="fas fa-download mr-2"></i>Export Report
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <div class="container mx-auto px-6 py-8">
        <!-- Stats Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div class="flex items-center">
                    <div class="p-3 rounded-full bg-blue-600 bg-opacity-20">
                        <i class="fas fa-file text-blue-500 text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <p class="text-gray-400 text-sm">Files Analyzed</p>
                        <p class="text-2xl font-bold" id="files-count">0</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div class="flex items-center">
                    <div class="p-3 rounded-full bg-green-600 bg-opacity-20">
                        <i class="fas fa-check-circle text-green-500 text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <p class="text-gray-400 text-sm">Findings</p>
                        <p class="text-2xl font-bold" id="findings-count">0</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div class="flex items-center">
                    <div class="p-3 rounded-full bg-yellow-600 bg-opacity-20">
                        <i class="fas fa-exclamation-triangle text-yellow-500 text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <p class="text-gray-400 text-sm">High Confidence</p>
                        <p class="text-2xl font-bold" id="high-confidence-count">0</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div class="flex items-center">
                    <div class="p-3 rounded-full bg-purple-600 bg-opacity-20">
                        <i class="fas fa-clock text-purple-500 text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <p class="text-gray-400 text-sm">Runtime</p>
                        <p class="text-2xl font-bold" id="runtime">00:00</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Progress Chart -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 class="text-lg font-semibold mb-4">Analysis Progress</h3>
                <canvas id="progressChart" height="200"></canvas>
            </div>
            
            <!-- Confidence Distribution -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 class="text-lg font-semibold mb-4">Confidence Distribution</h3>
                <canvas id="confidenceChart" height="200"></canvas>
            </div>
        </div>

        <!-- Tables Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Recent Findings -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 class="text-lg font-semibold mb-4">
                    <i class="fas fa-search mr-2"></i>Recent Findings
                </h3>
                <div class="overflow-y-auto max-h-96">
                    <table class="w-full" id="findings-table">
                        <thead class="text-gray-400 text-sm">
                            <tr>
                                <th class="text-left pb-2">Type</th>
                                <th class="text-left pb-2">Method</th>
                                <th class="text-left pb-2">Confidence</th>
                                <th class="text-left pb-2">Time</th>
                            </tr>
                        </thead>
                        <tbody id="findings-tbody">
                            <tr>
                                <td colspan="4" class="text-gray-500 text-center py-4">No findings yet</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Live Logs -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 class="text-lg font-semibold mb-4">
                    <i class="fas fa-terminal mr-2"></i>Live Logs
                </h3>
                <div class="bg-gray-900 rounded p-4 font-mono text-sm overflow-y-auto max-h-96" id="logs-container">
                    <div class="text-gray-500">Waiting for analysis to start...</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="bg-gray-800 border-t border-gray-700 mt-8">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between text-gray-400 text-sm">
                <div>StegAnalyzer v1.0 - Advanced Steganography Detection Framework</div>
                <div id="connection-status">
                    <i class="fas fa-circle text-green-400 mr-1"></i>Connected
                </div>
            </div>
        </div>
    </footer>

    <script src="/static/js/dashboard.js"></script>
</body>
</html>'''
        
        with open("templates/index.html", "w") as f:
            f.write(template_html)
    
    def _create_static_files(self):
        """Create CSS and JavaScript files"""
        
        # CSS
        css_content = '''
.finding-row:hover {
    background-color: rgba(55, 65, 81, 0.5);
}

.confidence-high {
    color: #ef4444;
}

.confidence-medium {
    color: #f59e0b;
}

.confidence-low {
    color: #10b981;
}

.log-entry {
    margin-bottom: 4px;
    padding: 2px 0;
}

.log-info {
    color: #3b82f6;
}

.log-warning {
    color: #f59e0b;
}

.log-error {
    color: #ef4444;
}

.log-success {
    color: #10b981;
}

#logs-container {
    scrollbar-width: thin;
    scrollbar-color: #4b5563 #1f2937;
}

#logs-container::-webkit-scrollbar {
    width: 6px;
}

#logs-container::-webkit-scrollbar-track {
    background: #1f2937;
}

#logs-container::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 3px;
}
'''
        
        with open("static/css/dashboard.css", "w") as f:
            f.write(css_content)
        
        # JavaScript
        js_content = '''
class DashboardApp {
    constructor() {
        this.ws = null;
        this.startTime = Date.now();
        this.charts = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.initCharts();
        this.startRuntimeTimer();
        this.setupEventHandlers();
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
            this.reconnectAttempts = 0;
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnect attempt ${this.reconnectAttempts}`);
                this.connectWebSocket();
            }, 2000 * this.reconnectAttempts);
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'session_start':
                this.handleSessionStart(data);
                break;
            case 'finding':
                this.handleFinding(data);
                break;
            case 'progress':
                this.handleProgress(data);
                break;
            case 'log':
                this.handleLog(data);
                break;
            case 'stats':
                this.handleStats(data);
                break;
        }
    }
    
    handleSessionStart(data) {
        document.getElementById('session-info').textContent = 
            `Session: ${data.session_id.slice(0, 8)}... | Target: ${data.target}`;
        this.startTime = Date.now();
    }
    
    handleFinding(data) {
        this.addFindingToTable(data);
        this.updateStats();
        this.updateCharts();
    }
    
    handleProgress(data) {
        this.updateProgressChart(data);
    }
    
    handleLog(data) {
        this.addLogEntry(data);
    }
    
    handleStats(data) {
        document.getElementById('files-count').textContent = data.files_analyzed || 0;
        document.getElementById('findings-count').textContent = data.total_findings || 0;
        document.getElementById('high-confidence-count').textContent = data.high_confidence || 0;
    }
    
    addFindingToTable(finding) {
        const tbody = document.getElementById('findings-tbody');
        
        // Remove "no findings" message
        if (tbody.children.length === 1 && tbody.children[0].cells.length === 1) {
            tbody.innerHTML = '';
        }
        
        const row = document.createElement('tr');
        row.className = 'finding-row transition-colors cursor-pointer';
        
        const confidenceClass = this.getConfidenceClass(finding.confidence);
        const time = new Date().toLocaleTimeString();
        
        row.innerHTML = `
            <td class="py-2 pr-4">
                <span class="px-2 py-1 bg-blue-600 bg-opacity-20 rounded text-xs">
                    ${finding.type || 'Unknown'}
                </span>
            </td>
            <td class="py-2 pr-4 text-gray-300">${finding.method || 'N/A'}</td>
            <td class="py-2 pr-4">
                <span class="${confidenceClass} font-semibold">
                    ${(finding.confidence * 100).toFixed(1)}%
                </span>
            </td>
            <td class="py-2 text-gray-400 text-sm">${time}</td>
        `;
        
        row.onclick = () => this.showFindingDetails(finding);
        
        tbody.insertBefore(row, tbody.firstChild);
        
        // Keep only last 50 findings
        while (tbody.children.length > 50) {
            tbody.removeChild(tbody.lastChild);
        }
    }
    
    addLogEntry(logData) {
        const logsContainer = document.getElementById('logs-container');
        
        // Remove "waiting" message
        if (logsContainer.children.length === 1 && 
            logsContainer.children[0].textContent.includes('Waiting')) {
            logsContainer.innerHTML = '';
        }
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${logData.level}`;
        
        const time = new Date().toLocaleTimeString();
        logEntry.innerHTML = `
            <span class="text-gray-500">[${time}]</span>
            <span class="ml-2">${logData.message}</span>
        `;
        
        logsContainer.appendChild(logEntry);
        
        // Keep only last 100 log entries
        while (logsContainer.children.length > 100) {
            logsContainer.removeChild(logsContainer.firstChild);
        }
        
        // Auto-scroll to bottom
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.7) return 'confidence-high';
        if (confidence >= 0.4) return 'confidence-medium';
        return 'confidence-low';
    }
    
    showFindingDetails(finding) {
        alert(`Finding Details:\\n\\nType: ${finding.type}\\nMethod: ${finding.method}\\nConfidence: ${(finding.confidence * 100).toFixed(1)}%\\nDetails: ${finding.details}`);
    }
    
    initCharts() {
        this.initProgressChart();
        this.initConfidenceChart();
    }
    
    initProgressChart() {
        const ctx = document.getElementById('progressChart').getContext('2d');
        this.charts.progress = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Completed', 'In Progress', 'Pending'],
                datasets: [{
                    data: [0, 0, 100],
                    backgroundColor: ['#10b981', '#3b82f6', '#6b7280'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#d1d5db' }
                    }
                }
            }
        });
    }
    
    initConfidenceChart() {
        const ctx = document.getElementById('confidenceChart').getContext('2d');
        this.charts.confidence = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Low (0-40%)', 'Medium (40-70%)', 'High (70%+)'],
                datasets: [{
                    label: 'Findings',
                    data: [0, 0, 0],
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: '#d1d5db' } },
                    y: { 
                        ticks: { color: '#d1d5db' },
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    updateProgressChart(data) {
        if (this.charts.progress) {
            this.charts.progress.data.datasets[0].data = [
                data.completed || 0,
                data.in_progress || 0,
                data.pending || 0
            ];
            this.charts.progress.update();
        }
    }
    
    updateCharts() {
        // Update confidence chart based on current findings
        const rows = document.getElementById('findings-tbody').children;
        const confidenceCounts = [0, 0, 0];
        
        for (let row of rows) {
            if (row.cells.length === 4) {
                const confidenceText = row.cells[2].textContent;
                const confidence = parseFloat(confidenceText) / 100;
                
                if (confidence < 0.4) confidenceCounts[0]++;
                else if (confidence < 0.7) confidenceCounts[1]++;
                else confidenceCounts[2]++;
            }
        }
        
        if (this.charts.confidence) {
            this.charts.confidence.data.datasets[0].data = confidenceCounts;
            this.charts.confidence.update();
        }
    }
    
    startRuntimeTimer() {
        setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            document.getElementById('runtime').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
    
    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (connected) {
            statusEl.innerHTML = '<i class="fas fa-circle text-green-400 mr-1"></i>Connected';
        } else {
            statusEl.innerHTML = '<i class="fas fa-circle text-red-400 mr-1"></i>Disconnected';
        }
    }
    
    setupEventHandlers() {
        document.getElementById('export-btn').onclick = () => {
            window.open('/export', '_blank');
        };
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new DashboardApp();
});
'''
        
        with open("static/js/dashboard.js", "w") as f:
            f.write(js_content)
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect_websocket(websocket)
        
        @self.app.get("/api/status")
        async def get_status():
            return {
                "status": "running",
                "session": self.current_session,
                "stats": self.session_stats,
                "uptime": time.time() - getattr(self, 'start_time', time.time())
            }
        
        @self.app.get("/api/sessions")
        async def get_sessions():
            sessions = await self.db.list_sessions(limit=10)
            return {"sessions": sessions}
        
        @self.app.get("/export")
        async def export_report():
            if not self.current_session:
                return {"error": "No active session"}
            
            # Generate and return report
            # This would integrate with the report generator
            return {"message": "Report generation not implemented yet"}
    
    async def connect_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            # Send initial data
            await websocket.send_text(json.dumps({
                "type": "init",
                "message": "Connected to StegAnalyzer Dashboard"
            }))
            
            # Keep connection alive
            while True:
                await websocket.receive_text()
                
        except WebSocketDisconnect:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                self.logger.debug(f"Failed to send message to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    async def start(self, session_id: str):
        """Start the dashboard server"""
        if not self.config.enabled:
            return
        
        self.current_session = session_id
        self.session_stats = {
            "files_analyzed": 0,
            "total_findings": 0,
            "high_confidence": 0,
            "start_time": time.time()
        }
        
        # Start server in background thread
        if not self.server_thread or not self.server_thread.is_alive():
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
        
        # Broadcast session start
        await self.broadcast_message({
            "type": "session_start",
            "session_id": session_id,
            "target": "Analysis Target",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Auto-open browser if configured
        if self.config.auto_open:
            try:
                webbrowser.open(self.url)
            except Exception as e:
                self.logger.debug(f"Could not open browser: {e}")
        
        self.logger.info(f"Dashboard started at {self.url}")
    
    def _run_server(self):
        """Run the FastAPI server"""
        try:
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="warning",  # Reduce uvicorn logging
                access_log=False
            )
        except Exception as e:
            self.logger.error(f"Dashboard server error: {e}")
    
    async def log_finding(self, finding: Dict[str, Any]):
        """Log a new finding to the dashboard"""
        await self.broadcast_message({
            "type": "finding",
            "finding": finding,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Update stats
        self.session_stats["total_findings"] += 1
        if finding.get("confidence", 0) >= 0.7:
            self.session_stats["high_confidence"] += 1
        
        await self.broadcast_message({
            "type": "stats",
            "stats": self.session_stats
        })
    
    async def log_progress(self, completed: int, in_progress: int, pending: int):
        """Log analysis progress"""
        await self.broadcast_message({
            "type": "progress",
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def log_message(self, level: str, message: str):
        """Log a general message"""
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in memory for dashboard
        self.real_time_logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.real_time_logs) > self.max_logs:
            self.real_time_logs = self.real_time_logs[-self.max_logs:]
        
        await self.broadcast_message({
            "type": "log",
            **log_entry
        })
    
    async def update_file_count(self, count: int):
        """Update the count of files analyzed"""
        self.session_stats["files_analyzed"] = count
        
        await self.broadcast_message({
            "type": "stats",
            "stats": self.session_stats
        })
    
    async def stop(self):
        """Stop the dashboard"""
        if not self.config.enabled:
            return
        
        # Broadcast session end
        await self.broadcast_message({
            "type": "session_end",
            "session_id": self.current_session,
            "final_stats": self.session_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close WebSocket connections
        for connection in self.active_connections.copy():
            try:
                await connection.close()
            except:
                pass
        
        self.active_connections.clear()
        self.current_session = None
        
        self.logger.info("Dashboard stopped")
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL"""
        return self.url
    
    def is_running(self) -> bool:
        """Check if dashboard is running"""
        return (self.server_thread is not None and 
                self.server_thread.is_alive() and 
                self.current_session is not None)