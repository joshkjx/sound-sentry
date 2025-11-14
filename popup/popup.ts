console.log("Popup Script initialised!");

interface ProcessedAudioData {
    // Depends on API return format
    // current fields are placeholders for mocking
    decision: string,
    confidence: number,
    chunksReceivedCount: number
}

interface AudioMetadata {
    framecount: number;
    duration: number;
    startTime: number;
    endTime: number;
}

interface ProcessedAudioMessage {
    type: 'PROCESSED_AUDIO';
    data: ProcessedAudioData;
    metadata: AudioMetadata;
}

interface ProcessingErrorMessage {
    type: 'PROCESSING_ERROR';
    error: string;
}


class PopupController {
    private port: chrome.runtime.Port | null = null;

    constructor() {
        this.init();
    }

    private init(): void {
        this.setupServiceWorkerConnection();
        // this.requestCurrentData();
    }

    private setupServiceWorkerConnection(): void {
        this.port = chrome.runtime.connect({name: 'popup-connection'});
        this.port.onMessage.addListener((message: ServiceWorkerMessage ) =>{
            if (!message){
                console.warn('Received null/undefined message');
            }

            if (!message.type) {
                console.warn('Received message without type:', message);
            }
            console.warn('🔔 MESSAGE RECEIVED IN POPUP:', message.type);
            this.handleServiceWorkerMessage(message)
        })
        this.port.onDisconnect.addListener(() =>{
            console.log('port between service worker and popup disconnected.')
            this.port = null;
        })
    }

    private handleServiceWorkerMessage(message: ServiceWorkerMessage){
        let messageData : ProcessedAudioData;
        if (message && message.type === "PROCESSED_AUDIO"){
            messageData = message.data;
            this.handleDataUpdate(messageData);
        } else {
            const fallback:ProcessedAudioData = {
                chunksReceivedCount:0,
                decision: "Not Applicable",
                confidence: 0.00
            }
            this.handleDataUpdate(fallback)
        }
    }

    private handleDataUpdate(data: ProcessedAudioData): void {
        // Update DOM elements with new data
        this.updateChunkCount(data.chunksReceivedCount);
        this.updateDecision(data.decision);
        this.updateConfidence(data.confidence);
    }

    private updateChunkCount(count: number): void {
        const element = document.getElementById('chunk-count');
        if (element) element.textContent = count.toString();
    }

    private updateDecision(decision: string){
        const element = document.getElementById('decision');
        if (element) element.textContent = decision;
    }

    private updateConfidence(confidence: number){
        const element = document.getElementById('confidence');
        if (element) element.textContent = confidence.toString();
    }
}


new PopupController();