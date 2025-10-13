// ============================================
// TYPE AUGMENTATION FOR MISSING BROWSER API
// ============================================

declare global {
    interface HTMLMediaElement { // This parent class is inherited by HTMLVideoElement,
        // but somehow TS doesn't recognise the captureStream() method despite the DOM supporting it.
        captureStream(): MediaStream;
    }
}

// This export is required for the augmentation to work
export {};


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

class AudioCapture {
    private mediaRecorder: MediaRecorder | null = null;
    private mediaStream: MediaStream | null = null;
    private currentVideo: HTMLVideoElement | null = null;
    private isCapturing: boolean = false;
    private isTabActive: boolean = !document.hidden;
    private port: chrome.runtime.Port | null = null;

    // Tracker for videos that have listeners attached
    private attachedVideos = new WeakSet<HTMLVideoElement>();

    // Track recording metadata
    private recordingStartTime: number = 0;
    private chunkCount: number = 0;

    constructor() {
        this.init();
    }

    private init(): void {
        this.setupPortConnection();
        this.setupVisibilityListener();
        this.initializeSiteCapture();
        this.setupCleanup();
    }

    private setupPortConnection(): void {
        this.connectPort();
    }

    private connectPort(): void {
        this.port = chrome.runtime.connect({name: 'audio-capture'});

        this.port.onMessage.addListener((message: ServiceWorkerMessage) => {
            this.handleServiceWorkerMessage(message);
        });

        this.port.onDisconnect.addListener(() => {
            console.log('Page-to-service-worker port disconnected, attempting reconnection...');
            this.port = null;

            setTimeout(() => {
                this.connectPort();
            }, 1000);
        });

        console.log('Port opened between main content and service worker');
    }

    private handleServiceWorkerMessage(message: ServiceWorkerMessage): void {
        switch (message.type){
            case 'PROCESSED_AUDIO':
                this.handleProcessedAudio(message.data, message.metadata);
                break;

            case 'PROCESSING_ERROR':
                console.error('Error during processing: ', message.error);
                break;

            default:
                console.log('Unknown message type: ', message);
        }
    }

    private setupVisibilityListener(): void {
        document.addEventListener('visibilitychange', () => {
            this.isTabActive = !document.hidden;

            if (this.isTabActive){
                console.log('Tab activity detected');


                // If there's a video, resume capture
                if (this.currentVideo && !this.currentVideo.paused) {
                    this.startCapture(this.currentVideo);
                }
            } else {
                console.log('Tab activity paused');

                this.stopCapture(false);
            }
        })
    }

    // ============================================
    // CLEANUP HANDLER
    // ============================================

    private setupCleanup(): void {
        // Clean up when page unloads
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    private cleanup(): void {
        console.log('Cleaning up AudioCapture setup...');
        this.stopCapture(true);
    }

    // ============================================
    // SITE-SPECIFIC INITIALIZATION
    // ============================================

    private initializeSiteCapture(): void {
        const hostname = window.location.hostname; // Getting hostname for the current page - used to check for supported sites

        if (hostname.includes('youtube.com')) {
            this.initYouTube();
        }
    }

    private initYouTube(): void {
        console.log('Initialising YouTube audio capture');

        // private method that checks if a video is on the page
        // Needs to be defined per supported website since the video css class used may differ.
        const checkVideo = (): void => {
            const video = document.querySelector<HTMLVideoElement>('video.html5-main-video');
            if (video) {
                this.attachVideoListeners(video); // if there is a video on the page, start to listen to it to get data
            }
        };

        //Check for navigation on YouTube, an SPA
        document.addEventListener('yt-navigate-finish', checkVideo);

        const observer = new MutationObserver(() => {
            if (!this.currentVideo) {
                checkVideo();
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        checkVideo();
    }

    // ============================================
    // VIDEO ELEMENT HANDLING
    // ============================================

    private attachVideoListeners(videoElement: HTMLVideoElement): void {
        // Prevent attaching redundant listeners
        if (this.attachedVideos.has(videoElement)) return;
        this.attachedVideos.add(videoElement);

        videoElement.addEventListener('play', () =>{
            if (this.isTabActive){
                this.startCapture(videoElement);
            }
        });

        videoElement.addEventListener('pause', () => {
            if (this.currentVideo === videoElement) {
                this.stopCapture();
            }
        });

        videoElement.addEventListener('ended', () => {
            if (this.currentVideo === videoElement) {
                this.stopCapture();
            }
        });

        // start capture if video is already playing in active tab
        if (!videoElement.paused && this.isTabActive) {
            this.startCapture(videoElement);
        }
    }

    // ============================================
    // AUDIO CAPTURE LOGIC
    // ============================================

    private startCapture(videoElement: HTMLVideoElement): void {
        //don't capture if inactive tab
        if (!this.isTabActive){
            return;
        }
        //do nothing if video already being captured
        if (this.isCapturing && this.currentVideo === videoElement){
            return;
        }
        //stop capture if it's a different video from the one currently being captured
        if (this.isCapturing && this.currentVideo !== videoElement){
            this.stopCapture();
        }

        //Safety check to make sure capture method is supported by browser
        if (!('captureStream' in videoElement)) {
            console.error('captureStream not supported in this browser');
            return;
        }

        try{
            // capture stream from videoElement (inbuilt method for the class)
            this.mediaStream = videoElement.captureStream();

            //create audio context and analyser (analyser is implicit to AudioContext class)
            if (!window.MediaRecorder){
                console.log('MediaRecorder not supported by browser');
                return;
            }

            const mimeType: string = 'audio/wav'
            this.mediaRecorder = new MediaRecorder(this.mediaStream, {
                mimeType: mimeType,
                audioBitsPerSecond: 128000 // 128kbps - random number I chose for now
            });

            this.recordingStartTime = Date.now();
            this.chunkCount = 0;

            this.mediaRecorder.ondataavailable = (event: BlobEvent) => { //The dataavailable event fires each time timeslice ms of media has been recorded.
                                                                               // The event is typed BlobEvent and contains the recorded media in data property
                this.handleAudioChunk(event.data);
            };

            // Handle errors
            this.mediaRecorder.onerror = (event: Event) => {
                console.error('MediaRecorder error:', event);
            };

            // Start recording with timeslice
            // Smaller timeslice = lower latency, more frequent chunks
            const CHUNK_DURATION_MS = 3000; // chunk duration in ms - this is 3 seconds for now. Each blob will contain 3s of audio data
            this.mediaRecorder.start(CHUNK_DURATION_MS);

            this.currentVideo = videoElement;
            this.isCapturing = true;

            console.log(`Audio Capture Started, Format = (${mimeType}, ${CHUNK_DURATION_MS}ms chunks`);

        } catch (error) {
            console.error('Failed to start capture: ', error);
        }
    }

    private stopCapture(clearState: boolean = true): void {
        this.isCapturing = false;

        //Stop MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.mediaRecorder = null;
        }

        // Stop media stream tracks
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Clear state if requested
        if (clearState) {
            this.currentVideo = null;
        }
        console.log('Audio Capture Stopped');
    }

    // ============================================
    // AUDIO CHUNK HANDLING
    // ============================================

    // Function to process each blob of 3s recorded data
    private handleAudioChunk(blob: Blob): void {
        if (blob.size === 0) {
            console.warn('Received empty audio chunk');
            return;
        }

        this.chunkCount++;
        const now = Date.now();
        const duration = now - this.recordingStartTime;

        console.log(`Audio chunk ${this.chunkCount}: ${blob.size} bytes, ${blob.type}`);

        if (this.port) { //sends chunk to service worker
            this.port.postMessage({
                type: 'AUDIO_CHUNK',
                blob: blob,
                timestamp: now,
                duration: duration
            });
        }
    }

    // ============================================
    // HANDLING OF PROCESSED DATA (PLACEHOLDER)
    // ============================================

    private handleProcessedAudio(
        data: ProcessedAudioData,
        metadata: AudioMetadata
    ) : void {
        const decision = data.decision;
        const confidence = data.confidence;
        const chunkCount = data.chunksReceivedCount;

    }
}

// ============================================
// INITIALIZATION
// ============================================

new AudioCapture();