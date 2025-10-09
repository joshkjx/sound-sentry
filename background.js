let recordingTabId = null;
let OFFSCREEN_DOC = "./offscreen/offscreen.html";

// Create the offscreen document
async function setupOffscreenDocument(path) {
  const offscreenUrl = chrome.runtime.getURL(path);
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [offscreenUrl]
  });

  if (!existingContexts.length) {
    await chrome.offscreen.createDocument({
      url: path,
      reasons: ['USER_MEDIA'],
      justification: 'To record tab audio using MediaRecorder.',
    });
  }
}

// Close the offscreen document
async function closeOffscreenDocument(path) {
  const offscreenUrl = chrome.runtime.getURL(path);
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [offscreenUrl]
  });

  if (existingContexts.length) {
    await chrome.offscreen.closeDocument();
  }
}

// Listener for messages from the popup and offscreen document
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Start recording from popup.js
  if (message.action === 'startRecording') {
    (async () => {

      await setupOffscreenDocument(OFFSCREEN_DOC);
      
      // Store the ID of the tab being recorded
      recordingTabId = message.tabId;
      
      // forward the stream ID to the offscreen document to start MediaRecorder
      const response = await chrome.runtime.sendMessage({
        action: 'startRecording',
        streamId: message.streamId
      });
      sendResponse(response);
    })();
    return true;
  }
  
  // Stop recording process
  else if (message.action === 'stopRecording') {
    (async () => {
      const response = await chrome.runtime.sendMessage({
        action: 'stopRecording'
      });
      sendResponse(response);
    })();
    return true;
  }
  
  // download audio
  else if (message.action === 'downloadAudio') {
    (async () => {
      const filename = `tab-audio-capture-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.webm`;
      await chrome.downloads.download({
        url: message.audioUrl,
        filename: filename,
        saveAs: true 
      });

      recordingTabId = null;
      await closeOffscreenDocument(OFFSCREEN_DOC);

      chrome.runtime.sendMessage({ action: 'recordingStopped' });
    })();
    return true;
  }
  
  //Status check from popup.js
  else if (message.action === 'checkRecordingStatus') {
    (async () => {
      const existingContexts = await chrome.runtime.getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
        documentUrls: [chrome.runtime.getURL(OFFSCREEN_DOC)]
      });
      
      let isRecording = false;
      if (existingContexts.length > 0) {
        // Ask the offscreen document for its actual MediaRecorder state
        const response = await chrome.runtime.sendMessage({ action: 'checkRecordingStatus' });
        isRecording = response ? response.isRecording : false;
      }
      sendResponse({ isRecording: isRecording });
    })();
    return true;
  }

  // error/cleanup
  else if (message.action === 'recordingStopped') {
    (async () => {
      recordingTabId = null;
      await closeOffscreenDocument(OFFSCREEN_DOC);
      // Forward the message to the popup for UI update
      chrome.runtime.sendMessage(message); 
    })();
    return true;
  }
});

// clean up if tab is closed
chrome.tabs.onRemoved.addListener(async (tabId) => {
  if (tabId === recordingTabId) {
    // Send a stop signal to the offscreen document
    // The offscreen.js onstop handler will then handle cleanup and download if possible, 
    // or just cleanup.
    try {
      await chrome.runtime.sendMessage({ action: 'stopRecording' });
    } catch (e) {
      // The offscreen document might already be closed if recording was stopped gracefully.
    }
    recordingTabId = null;
    await closeOffscreenDocument(OFFSCREEN_DOC);
    chrome.runtime.sendMessage({ action: 'recordingStopped', error: 'Tab closed.' });
  }
});