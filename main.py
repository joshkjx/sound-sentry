from predict_audio import predict

AUDIO_PATH = "dataset/8.wav"  # shld be fake
prob = predict(AUDIO_PATH)
print(f"Probability that '{AUDIO_PATH}' is fake: {prob:.3f}")
