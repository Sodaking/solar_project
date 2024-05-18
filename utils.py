from dotenv import load_dotenv
import os 

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document

load_dotenv()
upstage_api_key = os.getenv("UPSTAGE_API_KEY")


llm = ChatUpstage()

import requests
def extract_keyword(question):
    prompt_template = PromptTemplate.from_template(
    """
    사용자가 아래와 같은 질문을 물어봤을 때, 관련된 Wikipedia 검색에 가장 적합한 키워드 3개를 추출해줘.
    각각의 키워드는 매우 구체적이어야 하며, Wikipedia에서 잘 검색될 수 있도록 작성해줘.
    각 키워드는 한 줄로 작성하고, 쉼표로 구분해줘.
    예를 들면 다음과 같아. 
    Q : "로마 제국의 멸망 원인은 무엇인가요?"
    A : "로마 제국, 로마 제국 멸망 이유, 게르만족의 침입"
    Q : "서울대학교는 어디에 있나요?"
    A : "서울대학교, 서울대학교 위치, 서울대학교 캠퍼스"
    ---
    {question}
    """
    )
    chain = prompt_template | llm | StrOutputParser()
    result = chain.invoke({"question" : question})
    keywords = result.split(',')
    return keywords

def search_wikipedia(keyword):
    url = f"https://ko.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "format": "json"
    }
    response = requests.get(url, params=params).json()
    search_results = response['query']['search']
    if search_results:
        search_titles = [i["title"] for i in search_results]
        wikipedia_urls = [f"https://ko.wikipedia.org/wiki/{title.replace(' ', '_')}" for title in search_titles]
        return search_titles
    return None
def get_wikipedia_title(question):
    keywords = extract_keyword(question)
    # keyword = "연세대학교"
    print(keywords)
    if keywords:
        wikipedia_title = []
        for keyword in keywords[:1]:
            for title in search_wikipedia(keyword)[:1]:
                if title not in wikipedia_title:
                    wikipedia_title.append(title)
        return wikipedia_title
    return None
    
def retrieve_wikipedia_content(html_text, question):
    soup = BeautifulSoup(html_text, 'html.parser')
    elements = soup.find_all()
    sentences = []
    for element in elements:
        if element.text:
            a = element.text.split('.')
            for s in a:
                if s and len(s) > 5:
                    sentences.append(s)
    sample_text_list = sentences

    sample_docs = [Document(page_content=text, metadata={'ids': str(index)}) for index, text in enumerate(sample_text_list)]


    vectorstore = Chroma.from_documents(
        documents=sample_docs,
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
    )

    retriever = vectorstore.as_retriever()
    result_docs = retriever.invoke(question, top_k=5)
    result_texts = [doc.page_content for doc in result_docs]
    return result_texts

def mark_html_view(wiki_html, question):
    retrieve = retrieve_wikipedia_content(wiki_html, question)
    for i, text in enumerate(retrieve):
        print(text)
        wiki_html = wiki_html.replace(text, f"<mark>{text}</mark>")
    return wiki_html
