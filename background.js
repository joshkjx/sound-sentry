let recordingTabId = null;
let OFFSCREEN_DOC = "./offscreen/offscreen.html";

/**
 * Create the offscreen document
 *
 * @param {String} path - the url to use to create the offsreen document
 */
async function setupOffscreenDocument(path) {
  const offscreenUrl = chrome.runtime.getURL(path);
  // Retrieve all existing offscreen documents corresponding to this URL
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [offscreenUrl]
  });

  if (!existingContexts.length) {
    await chrome.offscreen.createDocument({
      url: path,
      reasons: ['USER_MEDIA'],
      justification: 'Setup an offscreen document to process audio information without interrupting user experience',
    });
  }
}

/**
 * Close the offscreen document
 * @param {String} path - the URL being emulated by the offscreen doc
 */
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

/**
 * Listener for messages from the popup and offscreen document
*/
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Start recording from popup.js
  if (message.action === 'startRecording') {
    (async () => {

      await setupOffscreenDocument(OFFSCREEN_DOC);
      
      // Store the ID of the tab being recorded
      recordingTabId = message.tabId;
      
      // forward the stream ID to the offscreen document to start MediaRecorder
      const response = await chrome.runtime.sendMessage({
        action: 'startRecordingProcess',
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
        action: 'stopRecordingProcess'
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

      const response = await chrome.runtime.sendMessage({ action: 'recordingStopped' });
      console.log(response);
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
        const response = await chrome.runtime.sendMessage({ action: 'checkRecordingStatusProcess' });
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
      const response = await chrome.runtime.sendMessage(message);
      console.log(response)
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
    const response = await chrome.runtime.sendMessage({ action: 'recordingStopped', error: 'Tab closed.' });
    console.log(response)
  }
});