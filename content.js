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
        this.audioOnlyStream = null;
        this.currentVideo = null;
        this.videoCurrentTime = 0; // Holds current timestamp for the playing video for tracking purposes when playback is interrupted
        this.isCapturing = false;
        this.isTabActive = !document.hidden;
        this.port = null;
        // Tracker for videos that have listeners attached
        this.attachedVideos = new WeakSet();
        // Track recording metadata
        this.recordingStartTime = 0;
        this.chunkCount = 0;
        this.getVideoMetadata = null; // Using an aliasing function that we can reassign during site-specific init.

        this.vidCaptureTimeout = null; // container for a timeout set by handleAudioChunk
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
            console.log("Service Worker Message Received")
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
                console.log("Audio Successfully Processed");
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

        // private method that resets capture state if YouTube SPA navigation is detected.
        const handleNavigation = () => {
            console.log('YouTube navigation detected. Resetting video state.');
            this.checkVideoWithRetry();
        }
        //Check for navigation on YouTube, an SPA
        document.addEventListener('yt-navigate-finish', handleNavigation);
        const observer = new MutationObserver(() => {
            if (!this.currentVideo) {
                this.checkVideoWithRetry();
            }
        });
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        this.getVideoMetadata = () => {
            return this.getYoutubeVideoMetadata();
        } // Set the metadata fetcher to the youtube implementation
    }

    getYoutubeVideoMetadata() {
        return {
            url: window.location.href,
            video_id: window.location.href.replace('https://www.youtube.com/watch?v=','YT-'), // extract youtube video id from url. Can use this as a unique identifier for videos.
            title: document.title.replace(' - YouTube', '') // Clean up YouTube suffix
        };
    }

    // ============================================
    // CAPTURE RESET AND RETRY LOGIC
    // ============================================
    resetCaptureState() {
        this.sendGraphReset();
        this.stopCapture(true);
        this.videoCurrentTime = 0;
        this.currentVideo = null;
        this.mediaStream = null;
        this.mediaRecorder = null;
        this.isCapturing = false;
        this.chunkCount = 0;
        this.recordingStartTime = 0;

        // Reset tracked video listeners
        this.attachedVideos = new WeakSet();

        console.log('AudioCapture state fully reset');
    }

    checkVideoWithRetry(retries = 3) {
        const video = document.querySelector('video.html5-main-video');
            if (video) {
                this.resetCaptureState();
                this.attachVideoListeners(video);
                console.log('New video element found and listeners reattached');
            } else if (retries > 0) {
                setTimeout(() => this.checkVideoWithRetry(retries - 1), 500); //half a second delay before reattempt to find video again
            } else {
                console.warn('Failed to find video element after navigation');
            }
    }
    // ============================================
    // VIDEO ELEMENT HANDLING
    // ============================================
    attachVideoListeners(videoElement) {
        // Prevent attaching redundant listeners
        if (this.attachedVideos.has(videoElement)) {
            if (!videoElement.paused && this.isTabActive) {
                this.startCapture(videoElement);
            }
            return;
        }
        this.attachedVideos.add(videoElement);
        videoElement.addEventListener('play', () => {
            if (this.isTabActive) {
                console.log("Play Event Received.");
                this.startCapture(videoElement);
            }
        });
        videoElement.addEventListener('pause', () => {
            if (this.currentVideo === videoElement) {
                console.log("Pause Event Received")
                this.stopCapture(false);
            }
        });
        videoElement.addEventListener('ended', () => {
            if (this.currentVideo === videoElement) {
                this.stopCapture();
            }
        });

        videoElement.addEventListener('seeked', () => { // fired when user finishes seeking
            if (this.currentVideo === videoElement) {
                console.log("videoNewTime is " + this.currentVideo.currentTime);
                console.log("stored time is: "+ this.videoCurrentTime);
                let videoNewTime = this.currentVideo.currentTime;
                if (videoNewTime < this.videoCurrentTime) {
                    console.log("Seek Backward Detected")
                    this.startCapture(videoElement);
                }
                else if (videoNewTime >= this.videoCurrentTime) {
                    console.log("Seek Forward Detected")
                    this.startCapture(videoElement);
                }
                else{
                    console.warn("Error occurred while handling seeking.");
                }
            }
        });
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
            console.log('Video has audio:', !!videoElement.audioTracks?.length); // Double negation to make it a bool for readable logs
            this.isCapturing = false;
            return;
        }
        console.log(`Captured ${audioTracks.length} audio track(s)`);
        if (!window.MediaRecorder) {
            console.log('MediaRecorder not supported by browser');
            return;
        }
        const streamAudioOnly = this.mediaStream.getAudioTracks();
        this.audioOnlyStream = new MediaStream(streamAudioOnly);

        this.recordingStartTime = Date.now();
        this.chunkCount = 0;
        this.isCapturing = true;
        this.currentVideo = videoElement;

        // Start first chunk
        this.startChunkRecording();

        console.log('Audio Capture Started, Format = (10000ms chunks)');
    }

    startChunkRecording() {
        if (!this.isCapturing) return;

        const startTime = Date.now();

        // Create NEW MediaRecorder for each chunk
        this.mediaRecorder = new MediaRecorder(this.audioOnlyStream, {
            mimeType: 'audio/webm;codecs=opus',
            audioBitsPerSecond: 128000
        });

        const chunks = [];
        const CHUNK_DURATION_MS = 4000;

        this.mediaRecorder.ondataavailable = (event) => {
            chunks.push(event.data);
        };

        this.mediaRecorder.onstop = () => {
            const blob = new Blob(chunks, {type: 'audio/webm;codecs=opus'});
            console.log('Complete chunk:', blob.size, 'bytes');
            this.handleAudioChunk(blob);
            // Clear any existing timeout before starting new recording
            if (this.chunkTimeout) {
                clearTimeout(this.chunkTimeout);
            }

            // Start next chunk
            this.startChunkRecording();
        };

        this.mediaRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event);
        };

        this.mediaRecorder.start(); // NO timeslice parameter
        console.log('Started recording chunk', ++this.chunkCount);

        // Clear any existing timeout before starting new recording
        if (this.chunkTimeout) {
            clearTimeout(this.chunkTimeout);
        }

        // Stop after 10 seconds
        this.chunkTimeout = setTimeout(() => {
            const actualDuration = Date.now() - startTime;
            console.log(`Stopping after ${actualDuration}ms, state: ${this.mediaRecorder.state}`);
            if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
            }
        }, CHUNK_DURATION_MS);
    }
    stopCapture(clearState = true) {
        this.isCapturing = false;
        if (this.chunkTimeout) clearTimeout(this.chunkTimeout);
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
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

    resumeCapture(videoElement, resumeTime){
        if (videoElement === this.currentVideo) {
                this.startCapture(videoElement);
            }
    }
    // ============================================
    // AUDIO CHUNK HANDLING
    // ============================================
    // Function to process each blob of 3s recorded data
    async handleAudioChunk(blob) {
        if (this.vidCaptureTimeout){ // If timeout is active, clear the timeout
            clearTimeout(this.vidCaptureTimeout);
            this.vidCaptureTimeout = null;
        }
        if (blob.size === 0) {
            console.warn('Received empty audio chunk');
            return;
        }
        this.chunkCount++;
        let videoPlaybackSeconds = 0;
        if (this.currentVideo) {
            if (this.currentVideo.readyState < 1) {
                console.warn('Video metadata not loaded yet');
                this.setVidCaptureTimeout();
                return;
            }
            this.videoPlaybackSeconds = this.currentVideo.currentTime;

        }
        const now = Date.now();
        const duration = now - this.recordingStartTime;
        const playbackTimestamp = this.videoPlaybackSeconds;
        const metadata = this.getVideoMetadata();

        console.log(`Audio chunk ${this.chunkCount}: ${blob.size} bytes, ${blob.type}`);
        if (this.port) { //sends chunk to service worker
            console.log('Attempting to post to Service Worker...');


            // Usage
            await this.verifyBlob(blob);  // CHECK THE BLOB
            const base64Data = await this.blobToBase64(blob);
            console.log('BEFORE SENDING - Base64 length:', base64Data.length);
            console.log('BEFORE SENDING - First 100 chars:', base64Data.substring(0, 100));

            console.log('Base64 length:', base64Data.length);

            this.port.postMessage({
                type: 'AUDIO_CHUNK',
                audioData: base64Data,
                mimeType: blob.type,
                timestamp: now,
                duration: duration,
                videoUrl: metadata.url,
                videoTitle: metadata.title,
                playbackTimestamp: playbackTimestamp
            });
            this.setVidCaptureTimeout(7000);
        }
    }
    // Helper function to get video title and url for more user-friendly experience
    setVidCaptureTimeout(time = 3000) {
        this.vidCaptureTimeout = setTimeout(() => {
            console.log("No audio chunk processed in the last 7 seconds, checking for video elements...");
            this.checkVideoWithRetry();
            this.vidCaptureTimeout = null;
        }, time);
    }

    // Helper func to convert blob to base64 for message passing
    async blobToBase64(blob) {
            return new Promise((resolve) => {
                const reader = new FileReader();

                reader.onloadend = () => {
                    const dataUrl = reader.result;
                    const base64Index = dataUrl.indexOf('base64,');
                    if (base64Index === -1) {
                        throw new Error('No base64 data found');
                    }
                    const base64 = dataUrl.substring(base64Index + 7); // Remove data:...;base64, prefix
                    resolve(base64);
                };
                reader.readAsDataURL(blob);
            });
    }

    async verifyBlob(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        console.log('Blob size before encoding:', blob.size);
        console.log('First 20 bytes before encoding:', Array.from(bytes.slice(0, 20)));
    }

    // ============================================
    // HANDLING OF PROCESSED DATA (PLACEHOLDER)
    // ============================================
    handleProcessedAudio(data, metadata) {
        const decision = data.decision;
        const confidence = data.confidence;
        const chunkCount = data.chunksReceivedCount;
        console.log("Decision: " + decision);
        console.log("Confidence: " + confidence);
        console.log("Chunks Received:" + chunkCount);
    }

    // ============================================
    // UTILITY FUNCTIONS
    // ============================================

    //Reset Graph in UI
    sendGraphReset() {
        if (this.port) {
            console.log("Sending GRAPH_RESET message to Service Worker.");
            this.port.postMessage({
                type: 'GRAPH_RESET'
            });
        }
    }

}
// ============================================
// INITIALIZATION
// ============================================
new AudioCapture();
