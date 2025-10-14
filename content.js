// ============================================
// TYPE AUGMENTATION FOR MISSING BROWSER API
// ============================================
// ============================================
// MAIN CLASS
// ============================================
class AudioCapture {
    constructor() {
        this.mediaRecorder = null;
        this.mediaStream = null;
        this.currentVideo = null;
        this.isCapturing = false;
        this.isTabActive = !document.hidden;
        this.port = null;
        // Tracker for videos that have listeners attached
        this.attachedVideos = new WeakSet();
        // Track recording metadata
        this.recordingStartTime = 0;
        this.chunkCount = 0;
        this.init();
    }
    init() {
        this.setupPortConnection();
        this.setupVisibilityListener();
        this.initializeSiteCapture();
        this.setupCleanup();
    }
    setupPortConnection() {
        this.connectPort();
    }
    connectPort() {
        this.port = chrome.runtime.connect({ name: 'audio-capture' });
        this.port.onMessage.addListener((message) => {
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
    handleServiceWorkerMessage(message) {
        switch (message.type) {
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
    setupVisibilityListener() {
        document.addEventListener('visibilitychange', () => {
            this.isTabActive = !document.hidden;
            if (this.isTabActive) {
                console.log('Tab activity detected');
                // If there's a video, resume capture
                if (this.currentVideo && !this.currentVideo.paused) {
                    this.startCapture(this.currentVideo);
                }
            }
            else {
                console.log('Tab activity paused');
                this.stopCapture(false);
            }
        });
    }
    // ============================================
    // CLEANUP HANDLER
    // ============================================
    setupCleanup() {
        // Clean up when page unloads
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }
    cleanup() {
        console.log('Cleaning up AudioCapture setup...');
        this.stopCapture(true);
    }
    // ============================================
    // SITE-SPECIFIC INITIALIZATION
    // ============================================
    initializeSiteCapture() {
        const hostname = window.location.hostname; // Getting hostname for the current page - used to check for supported sites
        if (hostname.includes('youtube.com')) {
            this.initYouTube();
        }
    }
    initYouTube() {
        console.log('Initialising YouTube audio capture');
        // private method that checks if a video is on the page
        // Needs to be defined per supported website since the video css class used may differ.
        const checkVideo = () => {
            const video = document.querySelector('video.html5-main-video');
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
    attachVideoListeners(videoElement) {
        // Prevent attaching redundant listeners
        if (this.attachedVideos.has(videoElement))
            return;
        this.attachedVideos.add(videoElement);
        videoElement.addEventListener('play', () => {
            if (this.isTabActive) {
                console.log("Play Event Received");
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
    startCapture(videoElement) {
        if (!this.isTabActive)
            return;
        if (this.isCapturing && this.currentVideo === videoElement)
            return;
        if (this.isCapturing && this.currentVideo !== videoElement) {
            this.stopCapture();
        }
        if (!('captureStream' in videoElement)) {
            console.error('captureStream not supported in this browser');
            return;
        }
        // capture stream from videoElement (inbuilt method for the class)
        this.mediaStream = videoElement.captureStream();
        // CHECK: Verify we have audio tracks
        const audioTracks = this.mediaStream.getAudioTracks();
        if (audioTracks.length === 0) {
            console.error('No audio tracks available in the captured stream');
            console.log('Video muted:', videoElement.muted);
            console.log('Video has audio:', !!videoElement.audioTracks?.length);
            return;
        }
        console.log(`Captured ${audioTracks.length} audio track(s)`);
        //create audio context and analyser (analyser is implicit to AudioContext class)
        if (!window.MediaRecorder) {
            console.log('MediaRecorder not supported by browser');
            return;
        }
        this.mediaRecorder = new MediaRecorder(this.mediaStream, {
            audioBitsPerSecond: 128000 // 128kbps - random number I chose for now
        });
        this.recordingStartTime = Date.now();
        this.chunkCount = 0;
        this.mediaRecorder.ondataavailable = (event) => {
            // The event is typed BlobEvent and contains the recorded media in data property
            this.handleAudioChunk(event.data);
        };
        // Handle errors
        this.mediaRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event);
        };
        // Start recording with timeslice
        // Smaller timeslice = lower latency, more frequent chunks
        const CHUNK_DURATION_MS = 3000; // chunk duration in ms - this is 3 seconds for now. Each blob will contain 3s of audio data
        this.mediaRecorder.start(CHUNK_DURATION_MS);
        this.currentVideo = videoElement;
        this.isCapturing = true;
        console.log(`Audio Capture Started, Format = (${CHUNK_DURATION_MS}ms chunks)`);
    }
    catch(error) {
        console.error('Failed to start capture: ', error);
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
    }
    stopCapture(clearState = true) {
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
    handleAudioChunk(blob) {
        if (blob.size === 0) {
            console.warn('Received empty audio chunk');
            return;
        }
        this.chunkCount++;
        const now = Date.now();
        const duration = now - this.recordingStartTime;
        const metadata = this.getVideoMetadata();
        console.log(`Audio chunk ${this.chunkCount}: ${blob.size} bytes, ${blob.type}`);
        if (this.port) { //sends chunk to service worker
            console.log('Attempting to post to Service Worker...');
            this.port.postMessage({
                type: 'AUDIO_CHUNK',
                blob: blob,
                timestamp: now,
                duration: duration,
                videoUrl: metadata.url,
                videoTitle: metadata.title
            });
        }
    }
    // Helper function to get video title and url for more user-friendly experience
    getVideoMetadata() {
        return {
            url: window.location.href,
            title: document.title.replace(' - YouTube', '') // Clean up YouTube suffix
        };
    }
    // ============================================
    // HANDLING OF PROCESSED DATA (PLACEHOLDER)
    // ============================================
    handleProcessedAudio(data, metadata) {
        const decision = data.decision;
        const confidence = data.confidence;
        const chunkCount = data.chunksReceivedCount;
    }
}
// ============================================
// INITIALIZATION
// ============================================
new AudioCapture();
