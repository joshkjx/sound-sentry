// ============================================
// TYPE DEFINITIONS
// ============================================
// ============================================
// MAIN CLASS
// ============================================
class AudioProcessor {
    constructor() {
        this.ports = new Set();
        this.popupPorts = new Set(); //Separate set for maintaining popup ports
        this.testingMode = false;
        this.latestAudioData = null;
        this.latestAudioMetadata = null;
        this.currentRecordingSessionChunksProcessed = 0;
        this.currentRecordingSessionRecords = [];
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
        if (port.name === 'popup-connection') {
            console.log('Successfully connected to popups');
            this.popupPorts.add(port);
            this.sendDataToPopup(port, {
                type: 'PROCESSED_AUDIO',
                data: this.latestAudioData,
                metadata: this.latestAudioMetadata
            });
            port.onDisconnect.addListener(() => {
                this.handlePopupPortDisconnection(port);
            });
        }
        else if (port.name === 'audio-capture') {
            this.ports.add(port);
            this.currentRecordingSessionStartTime = Date.now();
            port.onDisconnect.addListener(() => {
                this.handlePortDisconnection(port);
            });
            port.onMessage.addListener((message) => {
                console.log("Received Message from content script...");
                this.handleContentScriptMessage(message, port);
            });
        }
        else {
            console.log("Unknown port, disconnecting...");
            port.disconnect();
        }
    }
    handlePopupPortDisconnection(port) {
        this.popupPorts.delete(port);
    }
    handlePortDisconnection(port) {
        this.ports.delete(port);
        this.cleanUpOnDisconnect();
    }
    cleanUpOnDisconnect() {
        this.currentRecordingSessionChunksProcessed = 0;
        this.currentRecordingSessionRecords.length = 0; // TODO - check if this triggers a bug in the visualisation once that's up.
        this.currentRecordingSessionStartTime = null;
        this.latestAudioData = null;
        this.latestAudioMetadata = null;
    }
    // ============================================
    // MESSAGE HANDLING
    // ============================================
    handleContentScriptMessage(message, port) {
        switch (message.type) {
            case "AUDIO_CHUNK":
                this.handleAudioChunk(message.blob, message.timestamp, message.duration, message.videoUrl, message.videoTitle, message.playbackTimestamp ,port);
                break;
        }
    }
    async handleAudioChunk(blob, timestamp, duration, videoUrl, videoTitle, playbackTimestamp, port) {
        try {
            console.log('Service worker received audio chunk!');
            // Send audio data to API endpoint for processing
            const data = await this.sendToASRService(blob);
            // Build metadata for visualisation handling (maybe)
            const metadata = {
                framecount: 0, // TODO
                duration: duration,
                startTime: timestamp,
                endTime: timestamp + duration,
                playbackTimestamp: playbackTimestamp,
                videoTitle: videoTitle
            };
            console.log("test")
            console.log(metadata);
            let chunkVideoTitle = null;
            let chunkVideoUrl = null;
            if (videoTitle) {
                chunkVideoTitle = videoTitle;
            }
            if (videoUrl) {
                chunkVideoUrl = videoUrl;
            }
            this.setLatestData(data, metadata, chunkVideoTitle, chunkVideoUrl);
            this.sendProcessedAudio(port, data, this.latestAudioMetadata); // send audio back to content script on received port
            console.log('Broadcasting to popups...');
            this.broadcastToPopups(this.latestAudioData, this.latestAudioMetadata);
            // // console.log(this.latestAudioData)
            // console.log("this is metadata")
            // console.log(this.latestAudioMetadata);
            console.log('Service worker broadcast to popup');
        }
        catch (e) {
            const errorMsg = e instanceof Error ? e.message : String(e);
            this.sendProcessingError(port, errorMsg); // or send an error if failed.
        }
    }
    // Creates a record of the latest data. We can use this later to save to session storage. TODO: need to think of how to do this efficiently. After every chunk is handled?
    setLatestData(data, metadata, chunkVideoTitle, chunkVideoUrl) {
        this.latestAudioData = data;
        this.latestAudioMetadata = metadata;
        const currTime = Date.now();
        this.currentRecordingSessionChunksProcessed++;
        if (this.currentRecordingSessionChunksProcessed !== data.chunksReceivedCount) {
            console.warn('WARNING - chunks processed by service worker not equal to chunks received by API. Potential desync.');
        }
        let audioRecord = {
            timestamp: currTime,
            playbackTimestamp: metadata.playbackTimestamp,
            confidence: data.confidence,
            decision: data.decision,
        };
        if (chunkVideoUrl) {
            audioRecord.videoUrl = chunkVideoUrl;
        }
        if (chunkVideoTitle) {
            audioRecord.videoTitle = chunkVideoTitle;
        }
        this.currentRecordingSessionRecords.push(audioRecord);
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
    sendDataToPopup(port, message) {
        port.postMessage(message);
    }
    broadcastToPopups(data, metadata) {
        console.log("number of popup ports is: ", this.popupPorts.size);
        if (this.popupPorts.size === 0) {
            return;
        }
        this.popupPorts.forEach(port => {
            const message = {
                type: "PROCESSED_AUDIO",
                data: data,
                metadata: metadata
            };
            console.log('sending data to popups... Data is ', message.type);
            this.sendDataToPopup(port, message);
        });
    }
    // ============================================
    // MISCELLANEOUS HELPER FUNCTIONS
    // ============================================
    // Helper function to retrieve session data, used for continuity and for longer-horizon visualisation beyond the last received chunk.
    getRecordingSessionData() {
        return this.currentRecordingSessionRecords;
    }
    // Generates a unique session ID for session checking at the API side. No need to actually use this function if we find it unnecessary.
    generateRecordingSessionId(length = 8) {
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
// TEST CLASS
// ============================================
class MockApiTest {
    constructor() {
        this.callCount = 0;
        this.lastBlob = null;
        this.responseConfig = "SUCCESS";
        this.availableConfigs = new Array("SUCCESS", "ERROR", "TIMEOUT");
    }
    getResponse(blob) {
        return this.createResponse(blob);
    }
    getAvailableConfigs() {
        return this.availableConfigs;
    }
    setConfig(config) {
        this.setConfigHandler(config);
    }
    setConfigHandler(config) {
        if (this.availableConfigs.includes(config)) {
            this.responseConfig = config;
        }
        else {
            return;
        }
    }
    createResponse(blob) {
        this.lastBlob = blob;
        this.callCount++;
        if (blob.size === 0) {
            return Promise.reject(new Error('empty blob'));
        }
        switch (this.responseConfig) {
            case "SUCCESS":
                const successResponse = this.createSuccessResponse(blob);
                return Promise.resolve(successResponse);
            case "ERROR":
                const errorResponse = this.createErrorResponse();
                return Promise.reject(errorResponse);
            case "TIMEOUT":
                return new Promise((resolve, reject) => {
                    setTimeout(() => reject(new Error('Timeout')), 5000);
                });
            default:
                break;
        }
    }
    createSuccessResponse(blob) {
        const randomNumber = Math.random();
        const confidenceThreshold = 0.5;
        return {
            decision: randomNumber > confidenceThreshold ? "AI" : "Not AI",
            confidence: randomNumber,
            chunksReceivedCount: this.callCount
        };
    }
    createErrorResponse() {
        return new Error("Mock API Error Case");
    }
}
// ============================================
// INITIALIZATION
// ============================================
const processor = new AudioProcessor();
// Comment out below line if not testing with mock API
processor.activateTestMode(true, "SUCCESS");
