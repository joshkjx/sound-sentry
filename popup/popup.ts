interface DisplayData {
    chunksReceivedCount: number;
    decision: string;
    confidence: number;
}

class PopupController {
    private port: chrome.runtime.Port | null = null;

    constructor() {
        this.init();
    }

    private init(): void {
        this.setupServiceWorkerConnection();
        this.requestCurrentData();
    }

    private setupServiceWorkerConnection(): void {
        // Connect to service worker via port
        // Listen for data updates
    }

    private requestCurrentData(): void {
        // Send message to service worker requesting latest data
    }

    private handleDataUpdate(data: DisplayData): void {
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