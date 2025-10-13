// ============================================
// IMPORT TEST/MOCK CLASS
// ============================================
import { MockApiTest } from "./test_classes.js";


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

type ServiceWorkerMessage =
    | ProcessedAudioMessage
    | ProcessingErrorMessage;

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

    private testingUtil: MockApiTest;
    private testingMode: boolean = false;

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
        this.ports.add(port);
        port.onDisconnect.addListener(() => {
            this.handlePortDisconnection(port);
        })
        port.onMessage.addListener((message: ContentScriptMessage) => {
            this.handleContentScriptMessage(message, port)
        })
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
            // Send audio data to API endpoint for processing
            const data = await this.sendToASRService(blob);

            // Build metadata for visualisation handling (maybe)
            const metadata: AudioMetadata = {
                framecount: 0, // TODO
                duration: duration,
                startTime: timestamp,
                endTime: timestamp + duration
            };
            this.sendProcessedAudio(port, data, metadata); // send audio back to content script on received port
        } catch (e) {
            const errorMsg = e instanceof Error ? e.message: String(e);
            this.sendProcessingError(port,errorMsg); // or send an error if failed.
        }
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
// INITIALIZATION
// ============================================

const processor = new AudioProcessor();

// Comment out below line if not testing with mock API
processor.activateTestMode(true,"SUCCESS")