console.log("Popup Script initialised!");
class PopupController {
    constructor() {
        this.port = null;
        this.chartInstance = null;
        this.chartData = {
            labels: [],
            datasets: [{
                label: 'Probability Over Time',
                data: [],
                borderWidth: 2,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true,
                tension: 0.2
            }]
        };
        this.init();
    }
    init() {
        this.setupServiceWorkerConnection();
        this.initChart()
        // this.requestCurrentData();
    }
    setupServiceWorkerConnection() {
        this.port = chrome.runtime.connect({ name: 'popup-connection' });
        this.port.onMessage.addListener((message) => {
            if (!message) {
                console.warn('Received null/undefined message');
            }
            if (!message.type) {
                console.warn('Received message without type:', message);
            }
            console.warn('🔔 MESSAGE RECEIVED IN POPUP:', message.type);
            this.handleServiceWorkerMessage(message);
        });
        this.port.onDisconnect.addListener(() => {
            console.log('port between service worker and popup disconnected.');
            this.port = null;
        });
    }
    handleServiceWorkerMessage(message) {
        let messageData;
        let messageMetadata;
        if (message && message.type === "PROCESSED_AUDIO") {
            messageData = message.data;
            messageMetadata = message.metadata;
            this.handleDataUpdate(messageData, messageMetadata);
        }
        else if (message && message.type === "GRAPH_RESET") {
            console.log("Graph reset received. Clearing chart.");
            this.port.postMessage({ type: 'CONFIDENCE_WARNING', status: false });
            this.resetData();
        }
        else {
            const fallback = {
                chunksReceivedCount: 0,
                decision: "Not Applicable",
                confidence: 0.00
            };
            this.handleDataUpdate(fallback);
        }
    }
    handleDataUpdate(data, metadata) {
        if (metadata && metadata.videoTitle) {
            this.updateVideoTitle(metadata.videoTitle); 
        }

        //Decision Box Styling Logic
        const decisionPill = document.getElementById('decision-box');
        const decisionText = document.getElementById('decision');
        
        if (decisionPill && decisionText) {
            let bgColor = '#fff';
            let borderColor = '#333';
            let textColor = '#333';

            if (data.decision === 'AI') {
                bgColor = '#f8d7da';     
                borderColor = '#dc3545';
                textColor = '#dc3545';  
            } else if (data.decision === 'Not AI') {
                bgColor = '#d4edda';     
                borderColor = '#28a745';
                textColor = '#28a745';  
            }

            decisionPill.style.backgroundColor = bgColor;
            decisionPill.style.borderColor = borderColor;
            decisionText.style.color = textColor;
        }

        // Update DOM elements with new data
        this.updateChunkCount(data.chunksReceivedCount);
        this.updateDecision(data.decision);
        this.updateConfidence(data.confidence);
        this.updateConfidenceChart(data, metadata);
        this.checkWarningStatus();
    }
    updateVideoTitle(title) {
        const element = document.getElementById('video-title');
        if (element)
            element.textContent = title.toString();
    }
    updateChunkCount(count) {
        const element = document.getElementById('chunk-count');
        if (element)
            element.textContent = count.toString();
    }
    updateDecision(decision) {
        const element = document.getElementById('decision');
        if (element)
            element.textContent = decision;
    }
    updateConfidence(confidence) {
        const element = document.getElementById('confidence');
        const formattedConfidence = (confidence * 100).toFixed(2) + '%';
        
        if (element) {
            element.textContent = formattedConfidence;
        }
    }

    initChart() {
        const element = document.getElementById('confidence-chart');
        this.chartInstance = new Chart(element, {
            type: 'line',
            data: this.chartData,
            options: {
                responsive: true,
                animation: false, // disable animation for real-time updates
                scales: {
                    x: {
                        title: { display: true, text: 'Playback time' }
                    },
                    y: {
                        title: { display: true, text: 'Confidence' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    updateConfidenceChart(data, metadata) {
        if (!this.chartInstance) return;

        // const chunk = data.chunksReceivedCount;

        const playbackSeconds = metadata.playbackTimestamp
        const confidence = data.confidence;
        const decision = data.decision;

         // Format playback time into mm:ss
        const minutes = Math.floor(playbackSeconds / 60);
        const seconds = Math.floor(playbackSeconds % 60);
        const formattedPlayback = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        
        // add the new label and confidence point.
        this.chartData.labels.push(formattedPlayback);
        this.chartData.datasets[0].data.push(confidence);

        // limit chart length to 10 points
        if (this.chartData.labels.length > 10) {
            this.chartData.labels.shift();
            this.chartData.datasets[0].data.shift();
        }

        // store decision history
        if (!this.decisionHistory) this.decisionHistory = [];
        this.decisionHistory.push(decision);

        // Optional: limit chart & history length to 10 points
        if (this.chartData.labels.length > 10) {
            this.chartData.labels.shift();
            this.chartData.datasets[0].data.shift();
        }
        if (this.decisionHistory.length > 10) {
            this.decisionHistory.shift();
        }

        // Update the existing chart instance
        this.chartInstance.update();
    }

    checkWarningStatus() {
        const LAST_N_POINTS = 5;
        const REQUIRED_AI_COUNT = 2;

        const warningBox = document.getElementById('warning-box');
        if (!warningBox || !this.decisionHistory || this.decisionHistory.length < LAST_N_POINTS) {
            if (warningBox) warningBox.classList.add('hidden');
            return;
        }

         // Get the last N decisions
        const lastNDecisions = this.decisionHistory.slice(-LAST_N_POINTS);

        //Count how many are AI
        const aiCount = lastNDecisions.filter(d => d === 'AI').length;

        console.log(`Last ${LAST_N_POINTS} decisions: ${lastNDecisions.join(', ')}`);
        console.log(`AI count in last ${LAST_N_POINTS}: ${aiCount}`);

        if (aiCount >= REQUIRED_AI_COUNT) {
            // Show the warning box
            warningBox.classList.remove('hidden');
            this.port.postMessage({ type: 'CONFIDENCE_WARNING', status: true });
        }
         else {
            // Hide the warning box
            warningBox.classList.add('hidden');
            // this.port.postMessage({ type: 'CONFIDENCE_WARNING', status: false });
        }
        
    }

    resetData() {
        if (!this.chartInstance) return;

        // Clear all data arrays
        this.chartData.labels.length = 0;
        this.chartData.datasets[0].data.length = 0;

        // Clear decision history
        if (this.decisionHistory) {
            this.decisionHistory.length = 0;
        }

        // Update DOM elements to show 0
        this.updateChunkCount(0);
        this.updateDecision("N/A");
        this.updateConfidence(0);
        this.updateVideoTitle("Awaiting Video Data...");
        
        // Hide warning box
        const warningBox = document.getElementById('warning-box');
        if (warningBox) warningBox.classList.add('hidden');

        this.port.postMessage({ type: 'CONFIDENCE_WARNING', status: false });
        // Redraw chart
        this.chartInstance.update();
    }

}
new PopupController();
