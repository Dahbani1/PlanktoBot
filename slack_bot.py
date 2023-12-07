############# BOT IMPORTS ###########################
import os
#from slack_sdk.errors import SlackApiError
import slack
import os
############# AI IMPORTS ###########################
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


############# SETTINGS ###########################
SLACK_TOKEN_= ""
CHANNEL_ID_ = ""
#ADMIN_TOKEN_=  ""
#SIGNING_SECRET_= ""

client = slack.WebClient(token= SLACK_TOKEN_)  # Initialize a Web API client
BOT_ID = ''
#BOT_ID = client.auth_test()['user_id']


############# LISTEN TO MESSAGES IN A CHANNEL ###########################
def listen_to_channel(channel_id):
    # Get the last message in the channel
    result = client.conversations_history(channel=channel_id, limit=1)
    last_message = result["messages"][0]["text"]
    user = result["messages"][0]["user"]
    message_ts = result["messages"][0]["ts"]
    return last_message, user, message_ts, result

############# CREATE A THREAD & SEND MESSAGE ###########################
def send_message(channel_id, message, message_ts):
    try:
        result = client.chat_postMessage(channel=channel_id, text=message, thread_ts=message_ts)
    except slack.SlackApiError as e:
        print(f"Error: {e.response['error']}")


############# MAIN FUNCTION ###########################
if __name__ == "__main__":

    while True:
        last_message, user, message_ts, result = listen_to_channel(CHANNEL_ID_)

        if user != BOT_ID:
            if result['messages'][0]['blocks'][0]['elements'][0]['elements'][0]['user_id'] == BOT_ID:
                send_message(CHANNEL_ID_, f"Hi <@{user}>! Let me think... ")
                # if "hello" in last_message.lower():
                #     send_message(channel_id, f"Hi <@{user}>! Hello! How can I assist you." )
                # elif "Who are you" in last_message.lower():
                #     send_message(channel_id, f"Hi <@{user}>! I'm the , but I'm here to help!" )

                #Poser la question dans le content de "user"

                with open("faiss_store_openai.pkl", "rb") as f:
                    vectorStore = pickle.load(f)
                llm = ChatOpenAI(model_name='gpt-4', temperature=0)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore.as_retriever())
                # chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore_openAI.as_retriever())

                response = chain({"question": last_message}, return_only_outputs=True)
                send_message(CHANNEL_ID_, response)
