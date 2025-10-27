from predict_audio import predict
import time

AUDIO_PATH = "data/release-in-the-wild/33.wav"  # shld be fake
#start_time = time.time()
result = predict(AUDIO_PATH)
#end_time = time.time()

#elapsed = end_time - start_time

#print(f"\nPrediction completed in {elapsed:.3f} seconds")
print(f"Results for '{AUDIO_PATH}':")
print(result)
