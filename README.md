# SoundSentry Chrome Extension
Authors:
Angel Lee Xuan Xuan, Joshua Koh Jun Xiang, Yeo Zhuan Yu, Yim Sohyun

## Overview
SoundSentry is a lightweight Chrome extension that performs real-time detection of AI-generated speech in Youtube videos. Designed to protect users from synthetic audio misinformation, SoundSentry allows for accessible deployment directly in the Chrome browser.

The system builds on the DeepSonar framework, and has been adapted with with regularization and noise augmentation to allow for the robust real-time detection of AI-generated audio.

## Features
- Real-Time Detection: Continuously monitors audio from the active browser tab and classifies segments as real or AI-generated.

- DeepSonar-Based Architecture: Uses Top-K Activated Neurons (TKAN) extracted from a pyannote speaker embedding model.

- Visual Feedback: Displays a time-series chart of prediction probabilities, enabling intuitive understanding of risk.

## Loading the Chrome Extension
1. Clone or download this repository.
2. Navigate to ```chrome://extensions/``` and enable ```developer_mode```
3. Select ```load_unpacked``` and load in this repo's root folder.
4. Refresh, add and pin the extension to your chrome browser
5. Start using the extension to detect audio fakes!

## Miscellaneous
- ```const API_ENDPOINT``` is current disabled in ```service-worker.js```, contact owners for URL or replace with localhost instance. 
- Refer to README in ```back_end``` folder for more information on our architecture.
