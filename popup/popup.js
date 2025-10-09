document.addEventListener('DOMContentLoaded', () => {
  const startButton = document.getElementById('startRecording');
  const stopButton = document.getElementById('stopRecording');
  const statusDiv = document.getElementById('status');
  let isRecording = false;

  // Check initial state
  chrome.runtime.sendMessage({ action: 'checkRecordingStatus' }, (response) => {
    isRecording = response.isRecording;
    updateUI();
  });

  function updateUI() {
    if (isRecording) {
      startButton.style.display = 'none';
      stopButton.style.display = 'block';
      statusDiv.textContent = 'Recording...';
    } else {
      startButton.style.display = 'block';
      stopButton.style.display = 'none';
      statusDiv.textContent = 'Ready';
    }
  }

  // --- Start Recording ---
  startButton.addEventListener('click', async () => {
    if (isRecording) return;
    statusDiv.textContent = 'Requesting stream...';
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });

      await chrome.runtime.sendMessage({
        action: 'startRecording',
        streamId: streamId,
        tabId: tab.id
      });
      
      isRecording = true;
      updateUI();
    } catch (error) {
      console.error('Error starting capture:', error);
      statusDiv.textContent = 'Error: ' + error.message;
    }
  });

  // --- Stop Recording ---
  stopButton.addEventListener('click', () => {
    if (!isRecording) return;
    statusDiv.textContent = 'Stopping and Downloading...';

    chrome.runtime.sendMessage({ action: 'stopRecording' }, (response) => {
      if (response && response.success) {
        isRecording = false;
        updateUI();
        statusDiv.textContent = 'Download started!';
      } else {
        statusDiv.textContent = 'Error stopping recording.';
      }
    });
  });

  chrome.runtime.onMessage.addListener((message) => {
    if (message.action === 'recordingStopped') {
      isRecording = false;
      updateUI();
      statusDiv.textContent = message.error ? 'Error: ' + message.error : 'Download complete.';
    }
  });
});