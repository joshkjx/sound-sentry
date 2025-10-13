// ============================================
// IMPORT TEST/MOCK CLASS
// ============================================
import { MockApiTest } from "./test_classes";
// ============================================
// MAIN CLASS
// ============================================
class AudioProcessor {
    constructor() {
        this.ports = new Set();
        this.testingMode = false;
        this.init();
    }
    init() {
        this.setupPortListener();
    }
    // ============================================
    // PORT CONNECTION MANAGEMENT
    // ============================================
    setupPortListener() {
        chrome.runtime.onConnect.addListener((port) => {
            this.handlePortConnection(port);
        });
    }
    handlePortConnection(port) {
        this.ports.add(port);
        port.onDisconnect.addListener(() => {
            this.handlePortDisconnection(port);
        });
        port.onMessage.addListener((message) => {
            this.handleContentScriptMessage(message, port);
        });
    }
    handlePortDisconnection(port) {
        this.ports.delete(port);
    }
    // ============================================
    // MESSAGE HANDLING
    // ============================================
    handleContentScriptMessage(message, port) {
        switch (message.type) {
            case "AUDIO_CHUNK":
                this.handleAudioChunk(message.blob, message.timestamp, message.duration, port);
                break;
        }
    }
    async handleAudioChunk(blob, timestamp, duration, port) {
        try {
            // Send audio data to API endpoint for processing
            const data = await this.sendToASRService(blob);
            // Build metadata for visualisation handling (maybe)
            const metadata = {
                framecount: 0, // TODO
                duration: duration,
                startTime: timestamp,
                endTime: timestamp + duration
            };
            this.sendProcessedAudio(port, data, metadata); // send audio back to content script on received port
        }
        catch (e) {
            const errorMsg = e instanceof Error ? e.message : String(e);
            this.sendProcessingError(port, errorMsg); // or send an error if failed.
        }
    }
    // ============================================
    // API COMMUNICATION
    // ============================================
    async sendToASRService(blob) {
        // TODO: Convert blob to format needed by API
        // TODO: Make HTTP request to ASR endpoint
        // TODO: Parse and return response
        if (this.testingMode) { // Uses Mock Api if testing mode, otherwise falls back to API gateway.
            return this.testingUtil.getResponse(blob);
        }
        throw new Error('Not implemented');
    }
    // ============================================
    // RESPONSE HANDLING
    // ============================================
    sendProcessedAudio(port, data, metadata) {
        port.postMessage({
            type: 'PROCESSED_AUDIO',
            data: data,
            metadata: metadata
        });
    }
    sendProcessingError(port, error) {
        port.postMessage({
            type: 'PROCESSING_ERROR',
            error: error
        });
    }
    // ============================================
    // TESTING UTILITIES
    // ============================================
    activateTestMode(testMode = false, testConfig) {
        this.testingMode = testMode;
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
processor.activateTestMode(true, "SUCCESS");
