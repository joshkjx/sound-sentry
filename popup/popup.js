console.log("Popup Script initialised!");
class PopupController {
    constructor() {
        this.port = null;
        this.chartInstance = null;
        this.chartData = {
            labels: [],
            datasets: [{
                label: 'Confidence Over Time',
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
        if (message && message.type === "PROCESSED_AUDIO") {
            messageData = message.data;
            this.handleDataUpdate(messageData);
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
    handleDataUpdate(data) {
        // Update DOM elements with new data
        this.updateChunkCount(data.chunksReceivedCount);
        this.updateDecision(data.decision);
        this.updateConfidence(data.confidence);
        this.updateConfidenceChart(data);
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
        if (element)
            element.textContent = confidence.toString();
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
                        title: { display: true, text: 'Chunk Count' }
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

    updateConfidenceChart(data) {
        if (!this.chartInstance) return;

        const chunk = data.chunksReceivedCount;
        const confidence = data.confidence;

        // Add the new point
        this.chartData.labels.push(chunk);
        this.chartData.datasets[0].data.push(confidence);

        // Optional: limit chart length to 20 points
        if (this.chartData.labels.length > 20) {
            this.chartData.labels.shift();
            this.chartData.datasets[0].data.shift();
        }

        // Update the existing chart instance
        this.chartInstance.update();
    }
    // updateConfidenceChart(data) {
    //     const element = document.getElementById('confidence-chart');
    //     if (element)
    //         new Chart(element, {
    //         type: 'line',
    //         data: {
    //             labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
    //             datasets: [{
    //                 label: 'Confidence level',
    //                 data: [12, 19, 3, 5, 2, 3],
    //                 borderWidth: 1,
    //                 fill: true,
    //                 borderColor: 'rgb(75, 192, 192)',
    //                 tension: 0.1
    //             }]
    //         },
    //     });
    // }
}
new PopupController();
