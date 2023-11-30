import os
import time
from textwrap import dedent

import gradio as gr
import mdtex2html
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer

# fix timezone in Linux
os.environ["TZ"] = "Asia/Shanghai"
try:
    time.tzset()  # type: ignore # pylint: disable=no-member
except Exception:
    # Windows
    logger.warning("Windows, cant run time.tzset()")

# model_name = "THUDM/chatglm2-6b"  # 7x?G
model_name = "THUDM/chatglm-6b-int8"  # 3.9G

RETRY_FLAG = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
has_cuda = torch.cuda.is_available()
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True).half().cuda()
model = model.eval()
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """Copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/."""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


#===========添加submitBtn的相应为视频对应
def videochufa():
    print('运行videochufa')
    videostate.play() # ========这种对象,不能传参,只能用全局变量.
    videostate.autoplay=True
    videostate.value='test6.mp4'
    print(videostate.autoplay)
    from main9 import main
    # main('ffffff')
    videostate.update(value='result99999999.mp4')
    return videostate.update(value='result99999999.mp4') ########===========需要更新组件,要在返回函数里面写update,然后把值写上去.




def predict(
    RETRY_FLAG, input, chatbot, max_length, top_p, temperature, history, past_key_values
):
    try:
        chatbot.append((parse_text(input), ""))
    except Exception as exc:
        logger.error(exc)
        logger.debug(f"{chatbot=}")
        _ = """
        if chatbot:
            chatbot[-1] = (parse_text(input), str(exc))
            yield chatbot, history, past_key_values
        # """
        yield chatbot, history, past_key_values

    for response, history in model.stream_chat(
        tokenizer,
        input,
        history,
        past_key_values=past_key_values,
        return_past_key_values=True,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
    ):
        # print('得到的response',response[-1])
        chatbot[-1] = (parse_text(input), parse_text(response))
        past_key_values=None
        yield chatbot, history, past_key_values
    print('回答完所有之后函数的结果是',response)#=========在生成yield的循环之后获取即可.
    #==============这个函数后面加上数字人.


    #
    if 0:
        from main9 import main
        main(response)
        videostate.autoplay=True
    videochufa()







def reset_user_input():
    return gr.update(value="")


def reset_state():
    return [], [], None


# Delete last turn
def delete_last_turn(chat, history):
    if chat and history:
        chat.pop(-1)
        history.pop(-1)
    return chat, history


# Regenerate response
def retry_last_answer(
    user_input, chatbot, max_length, top_p, temperature, history, past_key_values
):
    if chatbot and history:
        # Removing the previous conversation from chat
        chatbot.pop(-1)
        # Setting up a flag to capture a retry
        RETRY_FLAG = True
        # Getting last message from user
        user_input = history[-1][0]
        # Removing bot response from the history
        history.pop(-1)

    yield from predict(
        RETRY_FLAG,  # type: ignore
        user_input,
        chatbot,
        max_length,
        top_p,
        temperature,
        history,
        past_key_values,
    )










with gr.Blocks(title="ChatGLM2-6B-int8", theme=gr.themes.Soft(text_size="sm")) as demo:
    with gr.Column(scale=4):
        with gr.Row():
            chatbot = gr.Chatbot()
            videostate =gr.Video(value='result99999999.mp4',width=400,height=400, autoplay=True)
            print(videostate,33333333333333)
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Input...",
                ).style(container=False)
                RETRY_FLAG = gr.Checkbox(value=False, visible=False)
            with gr.Column(min_width=32, scale=1):
                with gr.Row():
                    submitBtn = gr.Button("Submit", variant="primary")
                    deleteBtn = gr.Button("Delete last turn", variant="secondary")
                    retryBtn = gr.Button("Regenerate", variant="secondary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0,
                32768,
                value=8192,
                step=1.0,
                label="Maximum length",
                interactive=True,
            )
            top_p = gr.Slider(
                0, 1, value=0.85, step=0.01, label="Top P", interactive=True
            )
            temperature = gr.Slider(
                0.01, 1, value=0.95, step=0.01, label="Temperature", interactive=True
            )

    history = gr.State([])
    past_key_values = gr.State(None)

    user_input.submit(      # =======按回车相应
        predict,
        [
            RETRY_FLAG,
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        [chatbot, history, past_key_values],
        show_progress="full",
    )
    submitBtn.click(     #=========按submit按钮相应.
        predict,
        [
            RETRY_FLAG,
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        [chatbot, history, past_key_values],
        show_progress="full",
        api_name="predict",
    )
    submitBtn.click(reset_user_input, [], [user_input])


    

    # submitBtn.click(videochufa,[],[videostate]) ######不触发.













    emptyBtn.click(
        reset_state, outputs=[chatbot, history, past_key_values], show_progress="full"
    )

    retryBtn.click(
        retry_last_answer,
        inputs=[
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        # outputs = [chatbot, history, last_user_message, user_message]
        outputs=[chatbot, history, past_key_values],
    )
    deleteBtn.click(delete_last_turn, [chatbot, history], [chatbot, history])

demo.queue(concurrency_count=3, max_size=30).launch(debug=True,show_api=False)
