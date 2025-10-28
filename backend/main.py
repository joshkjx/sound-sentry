import os
import time
from predict_audio import AudioClassifier
from utils import DATA_DIR, DATASET

if __name__ == "__main__":
    classifier = AudioClassifier()

    for i in range(100):
        audio_file = os.path.join(DATA_DIR, DATASET, f"{i}.wav")

        if not os.path.exists(audio_file):
            print(f"Test file not found: {audio_file}")
            print("Please provide a valid audio file path")
        else:
            print(f"\nTesting on: {audio_file}\n")
            start_time = time.time()
            results = classifier.predict(audio_file)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"\nPrediction completed in {elapsed:.3f} seconds")

            gt = results.get('ground_truth', 'unknown')
            mp = results['mean_probability']
            conf = results['confidence']
            correct = 'Correct' if results.get('correct', False) else 'Wrong'
            print(f"{i}.wav: {results['overall']:4s} p={mp:.4f} GT={gt:10s} {correct} (conf: {conf:.4f})")

    # Cleanup
    classifier.cleanup()
