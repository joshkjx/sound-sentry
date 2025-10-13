// types.d.ts
declare global {
    interface HTMLMediaElement {
        captureStream(): MediaStream;
    }
}

export {}; // Required for global augmentation