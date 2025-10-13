console.log("Popup Script initialised!");
class PopupController {
    constructor() {
        this.port = null;
        this.init();
    }
    init() {
        this.setupServiceWorkerConnection();
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
}
new PopupController();
