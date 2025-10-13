// ============================================
// IMPORT TEST/MOCK CLASS
// ============================================


// ============================================
// TYPE DEFINITIONS
// ============================================

interface AudioChunkMessage {
    type: 'AUDIO_CHUNK';
    blob: Blob;
    timestamp: number;
    duration: number;
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

type ContentScriptMessage = AudioChunkMessage;

interface ProcessedAudioData {
    // Depends on API return format
    // TODO - fill in when API finalised
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

// ============================================
// MAIN CLASS
// ============================================

class AudioProcessor {
    private ports: Set<chrome.runtime.Port> = new Set();
    private popupPorts: Set<chrome.runtime.Port> = new Set(); //Separate set for maintaining popup ports

    private testingUtil: MockApiTest;
    private testingMode: boolean = false;
    private latestAudioData: ProcessedAudioData | null = null;
    private latestAudioMetadata: AudioMetadata | null = null;

    constructor() {
        this.init();
    }

    private init(): void {
        this.setupPortListener();
    }

    // ============================================
    // PORT CONNECTION MANAGEMENT
    // ============================================

    private setupPortListener(): void {
        chrome.runtime.onConnect.addListener((port: chrome.runtime.Port) => {
            this.handlePortConnection(port);
        });
    }

    private handlePortConnection(port: chrome.runtime.Port): void {
        if (port.name === 'popup-connection'){
            this.ports.add(port);
            this.sendDataToPopup(port, {
                type:'PROCESSED_AUDIO',
                data: this.latestAudioData,
                metadata: this.latestAudioMetadata
            });

        } else if (port.name === 'audio-capture') {
            this.ports.add(port);
            port.onDisconnect.addListener(() => {
                this.handlePortDisconnection(port);
            });
            port.onMessage.addListener((message: ContentScriptMessage) => {
                console.log("Received Message from content script...");
                this.handleContentScriptMessage(message, port)
            });
        } else {
            console.log("Unknown port, disconnecting...")
            port.disconnect();
        }
    }

    private handlePortDisconnection(port: chrome.runtime.Port): void {
        this.ports.delete(port);
    }

    // ============================================
    // MESSAGE HANDLING
    // ============================================

    private handleContentScriptMessage(
        message: ContentScriptMessage,
        port: chrome.runtime.Port
    ): void {
        switch (message.type) {
            case "AUDIO_CHUNK":
                this.handleAudioChunk(message.blob, message.timestamp, message.duration, port);
                break;
        }
    }

    private async handleAudioChunk(
        blob: Blob,
        timestamp: number,
        duration: number,
        port: chrome.runtime.Port
    ): Promise<void> {
        try {
            console.log('Service worker received audio chunk!');
            // Send audio data to API endpoint for processing
            const data = await this.sendToASRService(blob);

            // Build metadata for visualisation handling (maybe)
            this.latestAudioMetadata = {
                framecount: 0, // TODO
                duration: duration,
                startTime: timestamp,
                endTime: timestamp + duration
            };

            this.setLatestData(data);

            this.sendProcessedAudio(port, data, this.latestAudioMetadata); // send audio back to content script on received port

            console.log('Broadcasting to popups...')
            this.broadcastToPopups(this.latestAudioData, this.latestAudioMetadata);

        } catch (e) {
            const errorMsg = e instanceof Error ? e.message: String(e);
            this.sendProcessingError(port,errorMsg); // or send an error if failed.
        }
    }

    private setLatestData(data:ProcessedAudioData){
        this.latestAudioData = data;
    }

    // ============================================
    // API COMMUNICATION
    // ============================================


    private async sendToASRService(blob: Blob): Promise<ProcessedAudioData> {
        // TODO: Convert blob to format needed by API
        // TODO: Make HTTP request to ASR endpoint
        // TODO: Parse and return response

        if (this.testingMode){ // Uses Mock Api if testing mode, otherwise falls back to API gateway.
            return this.testingUtil.getResponse(blob);
        }

        throw new Error('Not implemented');
    }

    // ============================================
    // RESPONSE HANDLING
    // ============================================

    private sendProcessedAudio(
        port: chrome.runtime.Port,
        data: ProcessedAudioData,
        metadata: AudioMetadata
    ): void {
        port.postMessage({
            type: 'PROCESSED_AUDIO',
            data: data,
            metadata: metadata
        });
    }

    private sendProcessingError(
        port: chrome.runtime.Port,
        error: string
    ): void {
        port.postMessage({
            type: 'PROCESSING_ERROR',
            error: error
        });
    }

    private sendDataToPopup(port:chrome.runtime.Port, message: ProcessedAudioMessage): void{

        port.postMessage(message);
    }

    private broadcastToPopups(data:ProcessedAudioData, metadata: AudioMetadata): void{
        this.popupPorts.forEach(port =>{
            const message: ProcessedAudioMessage = {
                type: "PROCESSED_AUDIO",
                data: data,
                metadata: metadata
            }
            console.log('sending data to popups... Data is ', message.type);
            this.sendDataToPopup(port,message);
        })
    }

    // ============================================
    // TESTING UTILITIES
    // ============================================

    public activateTestMode(testMode: boolean = false,
                            testConfig: string) : void {
        this.testingMode = testMode

        if (!testMode) { // Defensive, early exit if not testing
            this.testingMode = false;
            return;
        }

        this.testingUtil = new MockApiTest();
        this.testingUtil.setConfig(testConfig);
    }

}


// ============================================
// TEST CLASS
// ============================================
class MockApiTest{
    private callCount: number = 0;
    private lastBlob: Blob | null = null;
    private responseConfig: string = "SUCCESS";
    private availableConfigs: Array<string>;

    constructor() {
        this.availableConfigs = new Array<string>("SUCCESS", "ERROR", "TIMEOUT");

    }


    public getResponse(blob: Blob): Promise<ProcessedAudioData>{
        return this.createResponse(blob)
    }

    public getAvailableConfigs(): Array<string> {
        return this.availableConfigs;
    }

    public setConfig(config:string){
        this.setConfigHandler(config);
    }

    private setConfigHandler(config:string) {
        if (this.availableConfigs.includes(config)){
            this.responseConfig = config;
        } else {
            return;
        }
    }

    private createResponse(blob: Blob): Promise<ProcessedAudioData>{
        this.lastBlob = blob
        this.callCount++;
        if (blob.size === 0){
            return Promise.reject(new Error('empty blob'))
        }
        switch (this.responseConfig){
            case "SUCCESS":
                const successResponse: ProcessedAudioData =  this.createSuccessResponse(blob);
                return Promise.resolve(successResponse);
            case "ERROR":
                const errorResponse: Error = this.createErrorResponse();
                return Promise.reject(errorResponse);
            case "TIMEOUT":
                return new Promise((resolve,reject) => {
                    setTimeout(() => reject(new Error('Timeout')),5000);
                });
            default:
                break;
        }

    }

    private createSuccessResponse(blob: Blob): ProcessedAudioData{
        const randomNumber: number = Math.random();
        const confidenceThreshold: number = 0.5;

        return {
            decision: randomNumber > confidenceThreshold ? "AI" : "Not AI",
            confidence: randomNumber,
            chunksReceivedCount: this.callCount
        }
    }

    private createErrorResponse(): Error {
        return new Error("Mock API Error Case");
    }
}

// ============================================
// INITIALIZATION
// ============================================

const processor = new AudioProcessor();

// Comment out below line if not testing with mock API
processor.activateTestMode(true,"SUCCESS")