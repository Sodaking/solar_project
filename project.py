from dotenv import load_dotenv
import os
import warnings
import gradio as gr

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import wikipediaapi
from utils import get_wikipedia_title, mark_html_view


load_dotenv()
upstage_api_key = os.getenv("UPSTAGE_API_KEY")

warnings.filterwarnings("ignore")

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='Solar (shlee@hcil.snu.ac.kr)',
    language='ko',
    # extract_format=wikipediaapi.ExtractFormat.WIKI
    extract_format=wikipediaapi.ExtractFormat.HTML
)
p_wiki = wiki_wiki.page("위키백과")
wiki_title = p_wiki.title
# wiki_html = """<h1><a href="https://ko.wikipedia.org/wiki/{wiki_title}">{wiki_title}</a></h1>""" + p_wiki.text
wiki_html = """<a href="https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8C%80%EB%AC%B8" class="mw-logo">
	<img class="mw-logo-icon" src="https://ko.wikipedia.org/static/images/icons/wikipedia.png" alt="" aria-hidden="true" height="50" width="50">
	<span class="mw-logo-container skin-invert">
		<img class="mw-logo-wordmark" alt="위키백과" src="https://ko.wikipedia.org/static/images/mobile/copyright/wikipedia-wordmark-ko.svg" style="width: 7.5em; height: 1.75em;">
		<img class="mw-logo-tagline" alt="" src="https://ko.wikipedia.org/static/images/mobile/copyright/wikipedia-tagline-ko.svg" width="120" height="13" style="width: 7.5em; height: 0.8125em;">
	</span>
</a>"""

llm = ChatUpstage()
chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)
chain = chat_with_history_prompt | llm | StrOutputParser()

def chat(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    response = chain.invoke({"message": message, "history": history_langchain_format})

    return response


def change_to_html(response):
    # Convert the chat response to HTML format
    html_response = f"""
    <h1>Chat Response</h1>
    <p>{response}</p>
    """
    return html_response

def update_html_view(chat_history, html_view):
    if chat_history and chat_history[-1][1] == None:

        wiki_titles = get_wikipedia_title(chat_history[-1][0])
        global wiki_html, wiki_title
        wiki_html = wiki_wiki.page(wiki_titles[0]).text
        wiki_title = wiki_wiki.page(wiki_titles[0]).title
        
        html_response = f"""
        <h1><a href="https://ko.wikipedia.org/wiki/{wiki_title}">{wiki_title}</a></h1>
        {wiki_html}
        """
        return gr.HTML(value=html_response)
    if chat_history and chat_history[-1][1] != None:
        html_response = f"""
        <h1><a href="https://ko.wikipedia.org/wiki/{wiki_title}">{wiki_title}</a></h1>
        {mark_html_view(wiki_html, chat_history[-1][0])}
        """
        return gr.HTML(value=html_response)
    return html_view


with gr.Blocks(css=".left-column { height: 90vh; } .right-column { height: 90vh; overflow-y: scroll; }") as demo:
    with gr.Row():
        with gr.Column(elem_classes="left-column"):
            chatbot = gr.ChatInterface(
                fn = chat,
                examples=[
                    "서울대학교는 어디에 있나요?",
                    "로마 제국의 멸망 원인은 무엇인가요?",
                    "chatgpt는 언제 출시되었나요?",
                ],
                title="Chat with Wiki",
            )
            chatbot.chatbot.height = 300
        with gr.Column(elem_classes="right-column"):
            html_view = gr.HTML(value=wiki_html)
        chatbot.chatbot.change(fn=update_html_view, inputs=[chatbot.chatbot, html_view], outputs=html_view)


if __name__ == "__main__":
    demo.launch()
