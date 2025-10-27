from predict_audio import predict

AUDIO_PATH = "data/release-in-the-wild/33.wav"  # shld be fake
prob = predict(AUDIO_PATH)
print(f"Probability that '{AUDIO_PATH}' is fake:\n")
print(prob)
