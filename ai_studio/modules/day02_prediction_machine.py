"""Day 2 module - Prediction Machine.

Unlocked by: solutions to Day 2 (supervised learning).
The notebook exports a bundle:  ai_studio/models/prediction_machine.joblib
    {"model": <fitted sklearn classifier>,
     "feature_names": [...4 columns...],
     "target_names": [...3 species...]}
"""
TITLE = "🐧 Prediction Machine"
REQUIRES = ["prediction_machine.joblib"]


def render(models_dir):
    import gradio as gr
    import joblib
    import pandas as pd

    bundle = joblib.load(models_dir / "prediction_machine.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    target_names = [str(t) for t in bundle["target_names"]]

    def predict(bill_len, bill_dep, flipper, mass):
        X = pd.DataFrame(
            [[bill_len, bill_dep, flipper, mass]], columns=feature_names
        )
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            return {n: float(p) for n, p in zip(target_names, probs)}
        pred = model.predict(X)[0]
        return {str(pred): 1.0}

    gr.Markdown(
        "Describe a penguin with the sliders — the classifier **you trained "
        "on Day 2** guesses its species, live."
    )
    with gr.Row():
        with gr.Column():
            s_bill_len = gr.Slider(30, 60, value=44, step=0.1,
                                   label="Bill length (mm)")
            s_bill_dep = gr.Slider(13, 22, value=17, step=0.1,
                                   label="Bill depth (mm)")
            s_flipper = gr.Slider(170, 235, value=200, step=1,
                                  label="Flipper length (mm)")
            s_mass = gr.Slider(2700, 6300, value=4200, step=50,
                               label="Body mass (g)")
        out = gr.Label(num_top_classes=3, label="Predicted species")

    inputs = [s_bill_len, s_bill_dep, s_flipper, s_mass]
    for s in inputs:
        s.change(predict, inputs, out)
