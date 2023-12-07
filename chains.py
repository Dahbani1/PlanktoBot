#from vect_store import vectorStore_openAI
import pickle

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


with open("faiss_store_openai.pkl", "rb") as f:
    vectorStore = pickle.load(f)
    
    
llm = ChatOpenAI(model_name='gpt-4', temperature=0)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore.as_retriever())
# chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore_openAI.as_retriever())

response = chain({"question": "What is a Planktoscope? "}, return_only_outputs=True)
print(response)
print(response['answer'])
print(response['sources'])

#chain({"question": "What materials were used to create Planktoscope? "}, return_only_outputs=True)
#chain({"question": "What are the benefits of the materials used to create Planktoscope? "}, return_only_outputs=True)
#chain({"question": "What are the steps to setup Planktoscope? "}, return_only_outputs=True)