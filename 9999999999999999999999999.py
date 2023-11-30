import gradio as gr
import os


def combine(a='1', b='2'):
    print('运行了.')
    return a + " " + b


def mirror(x):
    return x


with gr.Blocks() as demo:


    btn = gr.Button(value="Submit")
    btn.click(combine)

if __name__ == "__main__":
    demo.launch()
