import os
import time
from .predict_audio import AudioClassifier
from .utils import DATA_DIR, DATASET

if __name__ == "__main__":
    classifier = AudioClassifier()

    for i in range(10):
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

            gt = results['ground_truth'] if results['ground_truth'] is not None else "N/A"
            mp = results['mean_probability']
            conf = results['confidence']
            correct = results['correct']

            if correct is None:
                correct_str = "N/A"
            elif correct:
                correct_str = "Correct"
            else:
                correct_str = "Wrong"
            print(f"{i}.wav: {results['overall']:4s} p={mp:.4f} GT={gt:10s} {correct_str} (conf: {conf:.4f})")

    # Cleanup
    classifier.cleanup()
