import time
from pathlib import Path
from predict_audio import AudioClassifier
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Initializes the AudioClassifier and runs it on all audio files
# in the test_data directory, verifying predictions against
# folder names (fake/real).
def run_batch_test():
    print("Initializing AudioClassifier...")
    print("This may take a moment as models are loaded...")
    try:
        # Initialize the classifier once
        classifier = AudioClassifier()
    except Exception as e:
        print(f"FATAL: Could not initialize AudioClassifier: {e}")
        print("Please ensure all model files and dependencies are correct.")
        return

    print("Initialization complete.")

    # Configuration
    base_dir = Path("test_data")
    # Define the sub-folders (datasets) to process
    datasets = ["testing"]
    # Define the labels (ground truth)
    labels = ["fake", "real"]

    total_files = 0
    total_correct = 0
    total_no_speech = 0
    total_errors = 0

    all_ground_truth = []
    all_predictions = []

    start_batch_time = time.time()

    try:
        for dataset in datasets:
            for label in labels:

                # Construct the path, e.g., "test_data/testing/fake"
                current_dir = base_dir / dataset / label

                if not current_dir.is_dir():
                    print(f"\nSkipping: Directory not found: {current_dir}")
                    continue

                print(f"\n--- Processing folder: {current_dir} ---")

                # Get all .wav and .mp3 files in this directory
                audio_files = list(current_dir.glob("*.wav")) + \
                    list(current_dir.glob("*.mp3"))

                if not audio_files:
                    print("No .wav or .mp3 files found.")
                    continue

                # The expected result based on the folder name
                # We capitalize it to match the classifier's output ("Fake", "Real")
                expected_result = label.capitalize()

                for audio_file in tqdm(audio_files, desc=f"Processing {current_dir.relative_to(base_dir)}"):
                    total_files += 1
                    # print(f"\nTesting file: {audio_file.name}")

                    start_file_time = time.time()

                    try:
                        # Run prediction
                        # The predict method takes a string path
                        results = classifier.predict(str(audio_file))

                        elapsed_file = time.time() - start_file_time
                        prediction = results['overall']

                        # Verification logic
                        if prediction == "No Speech":
                            total_no_speech += 1
                            result_str = "NO SPEECH"
                            continue
                        elif prediction == expected_result:
                            total_correct += 1
                            result_str = "Correct"
                        else:
                            result_str = "WRONG"
                        all_ground_truth.append(expected_result)
                        all_predictions.append(prediction)

                        prob = results.get('mean_probability', 0.0)

                        # print(f"  -> Predicted: {prediction:10s} | "
                        #       f"Actual: {expected_result:10s} | "
                        #       f"Result: {result_str:10s} | "
                        #       f"Time: {elapsed_file:.2f}s (p={prob:.4f})")

                    except Exception as e:
                        total_errors += 1
                        # tqdm handles printing errors above the bar
                        print(
                            f"  -> ERROR processing file {audio_file.name}: {e}")
                        # Continue to the next file


    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user.")
    finally:
        # Final Summary
        end_batch_time = time.time()
        total_time = end_batch_time - start_batch_time

        # Calculate valid predictions (total files minus no-speech and errors)
        valid_predictions = total_files - total_no_speech - total_errors

        print("\n" + "="*40)
        print("          Batch Test Summary")
        print("="*40)
        print(f"Total time:       {total_time:.2f} seconds")
        print(f"Total files:      {total_files}")
        print(f"Files w/ errors:  {total_errors}")
        print(f"Files w/ no speech: {total_no_speech}")
        print(f"Valid predictions: {valid_predictions}")
        print(f"Total correct:    {total_correct}")

        if valid_predictions > 0:
            accuracy = (total_correct / valid_predictions) * 100
            print(f"\nAccuracy (Correct / Valid): {accuracy:.2f}%")
        else:
            print("\nNo valid predictions were made to calculate accuracy.")
            
        if valid_predictions > 0:
            print("\n" + "="*40)
            print("        Batch Test Confusion Matrix")
            print("="*40)

            # Ensure the labels are in the order you want ("Real", then "Fake")
            labels = ["Real", "Fake"]
            cm = confusion_matrix(
                all_ground_truth, all_predictions, labels=labels)

            print("                     Predicted Real   Predicted Fake")
            print(f"Actual Real          {cm[0][0]:<15} {cm[0][1]:<15}")
            print(f"Actual Fake          {cm[1][0]:<15} {cm[1][1]:<15}")

        # Cleanup hooks
        print("\nCleaning up classifier resources...")
        classifier.cleanup()
        print("Done.")


if __name__ == "__main__":
    run_batch_test()
