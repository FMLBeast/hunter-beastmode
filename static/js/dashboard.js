
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
        alert(`Finding Details:\n\nType: ${finding.type}\nMethod: ${finding.method}\nConfidence: ${(finding.confidence * 100).toFixed(1)}%\nDetails: ${finding.details}`);
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
