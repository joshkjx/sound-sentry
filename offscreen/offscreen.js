let audioStream;
let audioContext;
let audioSource;
let audioPort;
let isProcessing;
let analyser;

// Function to clean up all resources
function cleanup() {
  if (audioSource) {
    audioSource.disconnect();
    audioSource = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (audioStream) {
    audioStream.getTracks().forEach(track => track.stop());
    audioStream = null;
  }
}
/**
 * Initialises persistent port between offscreen and service worker for low-latency data passing
 */
function initializePort() {
    audioPort = chrome.runtime.connect({name: "audio-processor"});

    audioPort.onMessage.addListener((message) => {
        if (message.type === 'START_CAPTURE') {
            startAudioProcessing(message.streamId);
        } else if (message.type === 'STOP_CAPTURE') {
            stopAudioProcessing();
        }
    });

    audioPort.onDisconnect.addListener(() => {
        console.log('Audio port disconnected');
        stopAudioProcessing();
        audioPort = null;
    });
}

initializePort();

async function startAudioProcessing(streamId) {
    if (isProcessing) return;

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia
    }
}