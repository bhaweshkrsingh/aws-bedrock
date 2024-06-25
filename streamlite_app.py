# python -m streamlit run streamlit_app1.py

#import Essential dependencies
import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

OPENAI_API_KEY="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"
records_to_pull = 100
#function to load the vectordatabase
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
        return llm

#creating prompt template using langchain
def load_prompt():
        prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf answer "i donot know what the hell you are asking about"
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
        doc_txt = "\n\n".join(doc.page_content for doc in docs)
        print("\ndoc_txt: \n")
        print(doc_txt)
        return doc_txt


if __name__=='__main__':
        sl.header("welcome to the 📝PDF bot")
        sl.write("🤖 You can chat by Entering your queries ")
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()
        
        query=sl.text_input('Enter some text')
        
        
        if(query):
                #getting only the chunks that are similar to the query for llm to produce the output
                print("\nquery: \n")
                print(query)

                db = FAISS.from_documents(md_docs, embeddings)

# Retrievers
                retriever = db.as_retriever(search_kwargs={"k": records_to_pull})
                filtered_docs = retriever.get_relevant_documents(query)
                print(len(filtered_docs))



'''
                similar_embeddings=knowledgeBase.similarity_search_with_score(query, k=records_to_pull)
                print("\nsimilar_embeddings: \n")
                print(similar_embeddings)
                similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))
                #creating the chain for integrating llm,prompt,stroutputparser
                retriever = similar_embeddings.as_retriever(search_kwargs={"k": records_to_pull})

'''
                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                
                response=rag_chain.invoke(query)
                print("response: \n")
                print(response)
                sl.write(response)

               
        
        
        
        