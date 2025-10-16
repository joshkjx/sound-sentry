// ============================================
// TYPE DEFINITIONS
// ============================================

interface AudioChunkMessage {
    type: 'AUDIO_CHUNK';
    blob: Blob;
    timestamp: number;
    duration: number;
    videoUrl?: string;
    videoTitle?: string;
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

interface ProcessedAudioRecord{
    timestamp: number;
    confidence: number;
    decision: string;
    videoUrl?: string;
    videoTitle?: string;
}

interface RecordingSessionData{
    results: Array<ProcessedAudioRecord>;
    sessionStartTime: number;
    totalChunksProcessed: number;
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
    playbackTimestamp: number;
    videoTitle: string
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

    private currentRecordingSessionChunksProcessed: number = 0;
    private currentRecordingSessionStartTime: number;
    private currentRecordingSessionRecords: Array<ProcessedAudioRecord>;

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
            console.log('Successfully connected to popups')
            this.popupPorts.add(port);
            this.sendDataToPopup(port, {
                type:'PROCESSED_AUDIO',
                data: this.latestAudioData,
                metadata: this.latestAudioMetadata
            });
            port.onDisconnect.addListener(() => {
                this.handlePopupPortDisconnection(port)
            })

        } else if (port.name === 'audio-capture') {
            this.ports.add(port);
            this.currentRecordingSessionStartTime = Date.now();

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

    private handlePopupPortDisconnection(port: chrome.runtime.Port): void {
        this.popupPorts.delete(port);
    }


    private handlePortDisconnection(port: chrome.runtime.Port): void {
        this.ports.delete(port);
        this.cleanUpOnDisconnect();
    }

    private cleanUpOnDisconnect(): void {
        this.currentRecordingSessionChunksProcessed = 0;
        this.currentRecordingSessionRecords.length = 0; // TODO - check if this triggers a bug in the visualisation once that's up.
        this.currentRecordingSessionStartTime = null;
        this.latestAudioData = null;
        this.latestAudioMetadata = null;
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
                this.handleAudioChunk(message.blob, message.timestamp, message.duration, message.videoTitle, message.videoUrl, port);
                break;
        }
    }

    private async handleAudioChunk(
        blob: Blob,
        timestamp: number,
        duration: number,
        videoTitle: string,
        videoUrl: string,
        port: chrome.runtime.Port
    ): Promise<void> {
        try {
            console.log('Service worker received audio chunk!');
            // Send audio data to API endpoint for processing
            const data = await this.sendToASRService(blob);

            // Build metadata for visualisation handling (maybe)

            let chunkVideoTitle: string | null = null;
            let chunkVideoUrl: string | null = null;
            let chunkPlaybackTimestamp: number | null = null;

            if (videoTitle){
                chunkVideoTitle = videoTitle;
            }
            if (videoUrl){
                chunkVideoUrl = videoUrl;
            }

            const metadata = {
                framecount: 0, // TODO
                duration: duration,
                startTime: timestamp,
                endTime: timestamp + duration,
                playbackTimestamp: chunkPlaybackTimestamp,
                videoTitle: chunkVideoTitle
            };

            this.setLatestData(data,metadata,chunkVideoTitle,chunkVideoUrl);

            this.sendProcessedAudio(port, data, this.latestAudioMetadata); // send audio back to content script on received port

            console.log('Broadcasting to popups...')
            this.broadcastToPopups(this.latestAudioData, this.latestAudioMetadata);

        } catch (e) {
            const errorMsg = e instanceof Error ? e.message: String(e);
            this.sendProcessingError(port,errorMsg); // or send an error if failed.
        }
    }

    // Creates a record of the latest data. We can use this later to save to session storage. TODO: need to think of how to do this efficiently. After every chunk is handled?
    private setLatestData(data:ProcessedAudioData, metadata:AudioMetadata, chunkVideoTitle?: string | null, chunkVideoUrl?: string | null){
        this.latestAudioData = data;
        this.latestAudioMetadata = metadata;
        const currTime: number = Date.now();

        this.currentRecordingSessionChunksProcessed++;
        if (this.currentRecordingSessionChunksProcessed !== data.chunksReceivedCount){
            console.warn('WARNING - chunks processed by service worker not equal to chunks received by API. Potential desync.')
        }

        let audioRecord: ProcessedAudioRecord = {
            timestamp: currTime,
            confidence: data.confidence,
            decision: data.decision,
        };
        if (chunkVideoUrl){
            audioRecord.videoUrl = chunkVideoUrl;
        }
        if (chunkVideoTitle){
            audioRecord.videoTitle = chunkVideoTitle;
        }
        this.currentRecordingSessionRecords.push(audioRecord);

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
        console.log("number of popup ports is: ", this.popupPorts.size);
        if (this.popupPorts.size === 0) {
            return;
        }

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
    // MISCELLANEOUS HELPER FUNCTIONS
    // ============================================

    // Helper function to retrieve session data, used for continuity and for longer-horizon visualisation beyond the last received chunk.
    private getRecordingSessionData(): Array<ProcessedAudioRecord>{
        return this.currentRecordingSessionRecords;
    }

    // Generates a unique session ID for session checking at the API side. No need to actually use this function if we find it unnecessary.
    private generateRecordingSessionId(length: number = 8): string {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let result = '';
        const randomValues = new Uint8Array(length);
        crypto.getRandomValues(randomValues);

        for (let i = 0; i < length; i++) {
            result += chars[randomValues[i] % chars.length];
        }
        return result;
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