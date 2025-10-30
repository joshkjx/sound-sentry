// CLASS DECLARATION
export class MockApiTest {
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
