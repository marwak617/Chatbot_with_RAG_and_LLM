from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from ctransformers import AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
IF you don't know the answer, please just say that you don't know the answer.

Context: {context}
Question : {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA Retrieval for each vector stores
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


    return prompt

"""we used ctransformers because it's really fast , it helps in speeding 
instead of loading the original model  from llama and using it directly"""


def load_llm():
    return ChatGroq(
        groq_api_key="your groq key",
        model_name="llama3-70b-8192",  # or llama3-8b-8192, gemma-7b-it
        temperature=0.5,
        max_tokens=512
    )

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs={'k':2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt':prompt}
    )

    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device':'cpu'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response

##chainlit 

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await  msg.send()
    msg.content = "Hi, how can I help you ?!"
    await msg.update()
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")

    cl.user_session.set("chain", chain)
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True,answer_prefix_tokens=["FINAL","ANSWER"]
    )    
    cb.answer_reached=True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer +=f"\nNo Sources Found."
    await cl.Message(content=answer).send()

