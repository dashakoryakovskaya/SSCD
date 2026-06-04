import gradio as gr
import plotly.express as px
import pandas as pd
import logging
import numpy as np
import pandas as pd

from inference import predict

logging.basicConfig(level=logging.INFO)

    
def plotly_plot_video(video_path):
    data_emo = pd.DataFrame()
    data_emo['Emotion'] = ['😠 злость', '🤢 отвращение', '😨 страх', '😄 радость', '😐 нейтральность', '😢 печаль', '😲 удивление']
    data_per = pd.DataFrame()
    data_per['Personality'] = ['👐 открытость опыту', '💯 добросовестность', '🤗 доброжелательность', '🎉 экстраверсия', '🧘‍♀️ эмоциональная стабильность']
    try:
        pred_emo, pred_per =  predict(video_path)
        data_emo['Probability'] = pred_emo[0]
        data_per['Predict'] = pred_per[0]
        p_emo = px.bar(data_emo, x='Emotion', y='Probability', color="Probability")
        p_per = px.bar(data_per, x='Personality', y='Predict', color="Predict")
        return (
            p_emo, p_per
        )

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        data_emo['Probability'] = [0] * data_emo.shape[0]
        data_per['Predict'] = [0] * data_per.shape[0]
        p_emo = px.bar(data_emo, x='Emotion', y='Probability', color="Probability")
        p_per = px.bar(data_per, x='Personality', y='Predict', color="Predict")
        return (
            p_emo, p_per
        )

def create_demo_video():
    with gr.Blocks(theme='Nymbo/rounded-gradient', css=".gradio-container {background-color: #F0F8FF}", title="Emotion and Personality Detection") as demo:
        gr.Markdown("# Предсказание эмоций и персональных качеств")

        with gr.Row():
            video_input = gr.Video(
                sources=["upload", "webcam"],
                label="Record or Upload Video",
                format="mp4",
                interactive=True
            )
        with gr.Row():
            emo_plot = gr.Plot(label="Предсказание эмоций")
            per_plot = gr.Plot(label="Предсказание персональных качеств")

        video_input.change(fn=plotly_plot_video, inputs=video_input, outputs=[emo_plot, per_plot])
    return demo

def create_demo():
    audio = create_demo_video()
    demo = gr.TabbedInterface(
        [audio],
        ["Предсказание эмоций и персональных качеств"],
    )
    return demo
    

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()