import gradio as gr
import os


def video_identity(video):
    return 


with gr.Blocks() as demo:
       gr.Video()

if __name__ == "__main__":
    demo.queue(concurrency_count=3, max_size=30).launch(server_name='0.0.0.0',inbrowser=True,share=True,debug=True)