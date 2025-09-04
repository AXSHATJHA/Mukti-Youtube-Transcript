import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory # CORRECTED IMPORT
from youtube_transcript_api import YouTubeTranscriptApi

# --- Load Environment Variables ---
# This will load variables from a .env file in the same directory
load_dotenv()

# --- 1. FastAPI App and Request/Response Models ---

app = FastAPI(
    title="YouTube Video Chatbot API",
    description="An API that allows you to chat with a YouTube video.",
    version="1.0.0",
)

class ChatInput(BaseModel):
    """Schema for the input request body."""
    uri: str = Field(..., description="The full URL of the YouTube video.", example="https://www.youtube.com/watch?v=QsYGlZkevEg")
    input: str = Field(..., description="The user's question about the video.", example="What are the main announcements?")

class ChatOutput(BaseModel):
    """Schema for the final JSON response."""
    answer: str
    justification: str

# --- 2. Define the Tool for Getting Transcripts ---
@tool
def get_transcript(video_url: str) -> str:
    """
    Gets the transcript of a YouTube video from its full URL.
    This tool MUST be called first to get the video's content before answering any user questions.
    """
    print(f"--- Getting transcript for URL: {video_url} ---")
    try:
        # Extract video ID from the URL
        parsed_url = urlparse(video_url)
        if "youtube.com" in parsed_url.netloc:
            video_id = parse_qs(parsed_url.query).get("v", [None])[0]
        elif "youtu.be" in parsed_url.netloc:
            video_id = parsed_url.path.lstrip('/')
        else:
            raise ValueError("Invalid YouTube URL")

        if not video_id:
            return "Error: Could not extract video ID from the URL."

        # CORRECTED TRANSCRIPT FETCHING LOGIC
        # This is a more robust way to get the transcript that avoids the static method issue.
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try to find a manual transcript in English or Hindi, otherwise get a generated one.
        transcript = transcript_list.find_transcript(['en', 'hi'])
        fetched_transcript = transcript.fetch()

        full_transcript = " ".join([d['text'] for d in fetched_transcript])
        print("--- Transcript fetched successfully ---")
        return full_transcript
    except Exception as e:
        print(f"--- Error fetching transcript: {e} ---")
        return f"Error fetching transcript: {e}"

# --- 3. Configure the LLM and Bind Tools ---
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("The OPENAI_API_KEY environment variable is not set. Please create a .env file and add it.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_transcript])

# --- 4. Manage Chat History ---
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Gets the chat history for a given session ID (video URI).
    If no history exists, it creates a new one.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- 5. Create the Core Conversational Chain (FIXED) ---

def format_messages(input_dict: dict) -> list:
    """
    Helper function to combine chat history and new user messages into a single list
    that the LLM can process. This resolves the ValueError.
    """
    history = input_dict.get('history', [])
    new_messages = input_dict.get('messages', [])
    return history + new_messages

# This is the chain that will correctly process the output of RunnableWithMessageHistory
message_processing_chain = RunnableLambda(format_messages) | llm_with_tools

# We wrap this corrected chain with the history manager.
conversational_chain = RunnableWithMessageHistory(
    message_processing_chain,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="history",
)

# --- 6. Define the API Endpoint ---
@app.post("/chat", response_model=ChatOutput)
async def chat_with_video(chat_input: ChatInput):
    """
    Main endpoint to handle the chat conversation.
    """
    session_id = chat_input.uri
    user_question = chat_input.input
    history = get_session_history(session_id)
    has_transcript = any(isinstance(msg, ToolMessage) for msg in history.messages)
    
    prompt_messages = [
        HumanMessage(content=f"""
        You are a helpful assistant answering questions about a YouTube video.
        The video URL is: {chat_input.uri}

        Your task is to answer the user's question: "{user_question}"

        - If you do NOT have the transcript yet, your ONLY job is to call the `get_transcript` tool right now.
        - If you ALREADY have the transcript (from a previous turn), use it and the conversation history to answer the user's question.
        - Provide a concise answer and a justification for it based on the video's content.
        """)
    ]

    # This 'invoke' call now works correctly because the chain is properly structured.
    ai_response = conversational_chain.invoke(
        {"messages": prompt_messages},
        config={"configurable": {"session_id": session_id}}
    )

    if ai_response.tool_calls and not has_transcript:
        print("--- LLM decided to call the get_transcript tool ---")
        tool_outputs = []
        for tool_call in ai_response.tool_calls:
            tool_output = get_transcript.invoke(tool_call['args'])
            tool_outputs.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
            )
        
        print("--- Invoking LLM again with the transcript ---")
        # CORRECTED THE SECOND INVOCATION
        # We only need to pass the tool_outputs. RunnableWithMessageHistory handles the rest.
        final_response_message = conversational_chain.invoke(
            {"messages": tool_outputs},
            config={"configurable": {"session_id": session_id}}
        )
        ai_response = final_response_message

    final_llm = llm.with_structured_output(ChatOutput)
    final_prompt = f"""
    Based on the following conversation and video transcript, please answer the user's last question.

    Conversation History:
    {history.messages}

    User's Last Question: "{user_question}"
    
    Your Last Response was: "{ai_response.content}"

    Now, format this final response into a JSON with 'answer' and 'justification' fields.
    """
    
    final_structured_response = final_llm.invoke(final_prompt)
    return final_structured_response

# --- 7. Run the Application ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

