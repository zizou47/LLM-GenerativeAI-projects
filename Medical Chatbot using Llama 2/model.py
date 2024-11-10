from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores\\db_faiss"


custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""
# forces the model to focus only on what is provided in the context.
def set_custom_prompt():
    """
    Prompt template from QA retrieval from each vector store. 
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model='llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama', 
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}), # top 2 most relevant documents for answering the query
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2', #Converts documents and queries into embeddings
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
  
    llm = load_llm()  
    qa_prompt = set_custom_prompt()  
    qa = retrieval_qa_chain(llm, qa_prompt, db)  # where medical documents are stored as embeddings

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

## CHAINLIT 

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hello, welcome to the Medical Bot, How can I help you?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # Changed to get the session variable
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall({"query": message.content}, callbacks=[cb])

    answer = res["result"]
    sources = res.get("source_documents", [])  # Use .get() to avoid KeyError

    if sources:
        answer += "\nSources: " + str(sources)
    else:
        answer += "\nNo Sources Found"
        
    await cl.Message(content=answer).send()
