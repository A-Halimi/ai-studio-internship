"""Day 8 module - Language Lab.

Unlocked by: solutions to Day 8 (transformers & language models).
The notebook exports a settings token:  ai_studio/models/language_lab.json
    {"made_by": "...", "default_temperature": 0.9,
     "story_starters": ["...", ...]}
The transformer models themselves load from the Hugging Face cache
(pre-baked in the Docker image, auto-downloaded elsewhere).
"""
TITLE = "💬 Language Lab"
REQUIRES = ["language_lab.json"]

_PIPES = {}


def _pipe(task, model):
    """Lazy, cached pipeline loading so the Studio starts fast."""
    key = (task, model)
    if key not in _PIPES:
        from transformers import pipeline
        _PIPES[key] = pipeline(task, model=model)
    return _PIPES[key]


def render(models_dir):
    import json

    import gradio as gr

    cfg = json.loads((models_dir / "language_lab.json").read_text(
        encoding="utf-8"))
    starters = cfg.get("story_starters") or [
        "Late one night in the lab, a small robot woke up and said",
        "The last penguin on the iceberg looked at the satellite and",
    ]
    default_temp = float(cfg.get("default_temperature", 0.9))

    def mood(text):
        if not text.strip():
            return {}
        res = _pipe("text-classification",
                    "distilbert-base-uncased-finetuned-sst-2-english")(text)[0]
        other = "NEGATIVE" if res["label"] == "POSITIVE" else "POSITIVE"
        return {res["label"]: res["score"], other: 1 - res["score"]}

    def write_story(prompt, temperature, length):
        if not prompt.strip():
            return "Give me a story starter first!"
        gen = _pipe("text-generation", "gpt2")
        out = gen(prompt, max_new_tokens=int(length), do_sample=True,
                  temperature=float(temperature), top_p=0.95,
                  pad_token_id=50256)
        return out[0]["generated_text"]

    def guess_word(sentence):
        if "___" not in sentence:
            return {"Type a sentence with ___ for the blank": 1.0}
        fm = _pipe("fill-mask", "distilbert-base-uncased")
        masked = sentence.replace("___", fm.tokenizer.mask_token, 1)
        return {r["token_str"].strip(): float(r["score"])
                for r in fm(masked)[:5]}

    def sort_text(text, labels_csv):
        labels = [l.strip() for l in labels_csv.split(",") if l.strip()]
        if not text.strip() or len(labels) < 2:
            return {}
        zs = _pipe("zero-shot-classification",
                   "typeform/distilbert-base-uncased-mnli")
        res = zs(text, candidate_labels=labels)
        return {l: float(s) for l, s in zip(res["labels"], res["scores"])}

    gr.Markdown(
        "Four language superpowers from Day 8 — all running on the "
        "transformer models you explored."
    )
    with gr.Tabs():
        with gr.Tab("😊 Mood Meter"):
            t = gr.Textbox(label="Write a sentence",
                           placeholder="This internship is awesome!")
            o = gr.Label(label="Detected mood")
            t.submit(mood, t, o)
            gr.Button("Read the mood").click(mood, t, o)

        with gr.Tab("📖 Story Writer"):
            p = gr.Dropdown(choices=starters, value=starters[0],
                            allow_custom_value=True,
                            label="Story starter (pick one or write your own)")
            with gr.Row():
                temp = gr.Slider(0.2, 1.5, value=default_temp, step=0.05,
                                 label="Temperature (creativity)")
                length = gr.Slider(20, 120, value=60, step=10,
                                   label="Words to add (approx.)")
            story = gr.Textbox(label="GPT-2 continues...", lines=8)
            gr.Button("Write! ✍️", variant="primary").click(
                write_story, [p, temp, length], story)

        with gr.Tab("🕳️ Guess the Word"):
            s = gr.Textbox(label="Sentence with a blank (use ___)",
                           value="Students in the lab love to ___ every day.")
            o2 = gr.Label(label="The model's top guesses")
            s.submit(guess_word, s, o2)
            gr.Button("Fill the blank").click(guess_word, s, o2)

        with gr.Tab("🗂️ Sorting Hat"):
            txt = gr.Textbox(label="Any text",
                             value="The team trained a neural network on GPUs.")
            labs = gr.Textbox(label="Categories (comma-separated)",
                              value="science, sports, cooking, music")
            o3 = gr.Label(label="Where it belongs")
            gr.Button("Sort it!").click(sort_text, [txt, labs], o3)
