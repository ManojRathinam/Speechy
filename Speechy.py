from faster_whisper import WhisperModel
import gradio as gr
import random
import os

# Load model
model = WhisperModel("base", compute_type="int8")

# Cache for sentence state
sentence_cache = {"current": ""}

# Load sentences from file
def load_sentences(age_group):
    file_mapping = {
        "6-8": "sentences_6_8.txt",
        "9-11": "sentences_9_11.txt",
        "12-13": "sentences_12_13.txt"
    }
    try:
        with open(file_mapping[age_group], "r") as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return []

# Set sentence for user to read
def set_sentence(age_group):
    sentences = load_sentences(age_group)
    if not sentences:
        return "No sentences found for this age group."
    chosen = random.choice(sentences)
    sentence_cache["current"] = chosen
    return chosen

# Evaluate pronunciation
def evaluate(audio):
    expected_text = sentence_cache.get("current", "")
    if not expected_text:
        return "", "Please select an age group and sentence first.", 0

    # Transcribe audio
    segments, _ = model.transcribe(audio)
    transcript = " ".join(segment.text for segment in segments)

    # Accuracy check
    accuracy = get_accuracy(transcript, expected_text)
    feedback = f"You said: \"{transcript}\"\nExpected: \"{expected_text}\"\nAccuracy Score: {accuracy:.2f}%"
    return transcript, feedback, accuracy

# Accuracy calculation
def get_accuracy(transcript, expected_text):
    if not transcript or not expected_text:
        return 0
    t_words = transcript.lower().split()
    e_words = expected_text.lower().split()
    correct = sum(1 for t, e in zip(t_words, e_words) if t == e)
    return (correct / len(e_words)) * 100 if e_words else 0

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 🗣️Speechy - Pronunciation Practice for Kids")
    age = gr.Dropdown(choices=["6-8", "9-11", "12-13"], label="Choose Age Group")
    sentence_display = gr.Textbox(label="Read this sentence", interactive=False)
    get_sentence_btn = gr.Button("Get Sentence")
    get_sentence_btn.click(fn=set_sentence, inputs=age, outputs=sentence_display)

    audio_input = gr.Audio(type="filepath", label="Speak Now")
    transcript_out = gr.Textbox(label="Transcript")
    feedback_out = gr.Textbox(label="Feedback")
    accuracy_out = gr.Number(label="Accuracy Score (%)")

    evaluate_btn = gr.Button("Evaluate")
    evaluate_btn.click(fn=evaluate, inputs=audio_input, outputs=[transcript_out, feedback_out, accuracy_out])

app.launch(server_name="0.0.0.0", server_port=8080)
