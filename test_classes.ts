// TYPE DEFINITIONS
interface ProcessedAudioData {
    // Depends on API return format
    // TODO - fill in when API finalised
    // current fields are placeholders for mocking
    decision: string,
    confidence: number,
    chunksReceivedCount: number
}

// CLASS DECLARATION
export class MockApiTest{
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