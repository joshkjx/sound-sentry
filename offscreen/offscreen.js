let mediaRecorder;
let recordedChunks = [];
let audioStream;
let audioContext;
let audioSource;

// Function to clean up all resources
function cleanup() {
  if (mediaRecorder) {
    mediaRecorder.stop();
    mediaRecorder = null;
  }
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
  recordedChunks = [];
}

// Function: starts the recording
async function startRecording(streamId) {
  cleanup(); 

  try {

    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId
        }
      },
      video: false
    });

    audioContext = new AudioContext();
    audioSource = audioContext.createMediaStreamSource(audioStream);
    audioSource.connect(audioContext.destination);

    mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    //this stops the recording
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'audio/webm' });
      const url = URL.createObjectURL(blob);

      chrome.runtime.sendMessage({
        action: 'downloadAudio',
        audioUrl: url
      });
      
      cleanup();
    };

    mediaRecorder.start();
    return true;
  } catch (error) {
    console.error('Recording error in offscreen document:', error);
    chrome.runtime.sendMessage({ 
      action: 'recordingStopped', 
      error: 'Failed to start or process audio.' 
    });
    cleanup();
    return false;
  }
}


chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'startRecording') {
    startRecording(message.streamId).then(sendResponse);
    return true; 
  } else if (message.action === 'stopRecording') {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      sendResponse({ success: true });
    } else {
      sendResponse({ success: false, error: 'No active recording.' });
    }
  } else if (message.action === 'checkRecordingStatus') {
     sendResponse({ isRecording: mediaRecorder && mediaRecorder.state === 'recording' });
  }
});