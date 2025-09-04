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
from langchain_community.chat_message_histories import ChatMessageHistory
from youtube_transcript_api import YouTubeTranscriptApi

# --- Load Environment Variables ---
load_dotenv()

# --- 1. FastAPI App and Request/Response Models ---
app = FastAPI(
    title="YouTube Video Chatbot API",
    description="An API that allows you to chat with a YouTube video.",
    version="1.0.0",
)

class ChatInput(BaseModel):
    uri: str = Field(..., description="The full URL of the YouTube video.", example="https://www.youtube.com/watch?v=QsYGlZkevEg")
    input: str = Field(..., description="The user's question about the video.", example="What are the main announcements?")

class ChatOutput(BaseModel):
    answer: str
    justification: str

# --- 2. Define the Tool for Getting Transcripts ---
@tool
def get_transcript(video_url: str) -> str:
    """
    Gets the transcript of a YouTube video from its full URL.
    Robustly tries: manual EN/HI -> generated EN/HI -> translated-to-EN -> fallback static method.
    Returns only the transcript text. If it fails, returns a string that starts with "ERROR::".
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
            return "ERROR:: Invalid YouTube URL"

        if not video_id:
            return "ERROR:: Could not extract video ID from the URL."

        # Try multiple strategies to obtain a transcript
        fetched_transcript = None
        transcript = None
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except Exception as e:
            return f"ERROR:: Unable to list transcripts: {e}"

        # 1) Manual transcript in EN/HI
        try:
            transcript = transcript_list.find_transcript(['en', 'hi'])
        except Exception:
            transcript = None

        # 2) Generated transcript in EN/HI
        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'hi'])
            except Exception:
                transcript = None

        # 3) Any transcript translated to EN
        if transcript is None:
            try:
                for t in transcript_list:
                    try:
                        transcript = t.translate('en')
                        if transcript:
                            break
                    except Exception:
                        continue
            except Exception:
                transcript = None

        if transcript is not None:
            try:
                fetched_transcript = transcript.fetch()
            except Exception as e:
                fetched_transcript = None

        # 4) Final fallback: static helper
        if fetched_transcript is None:
            try:
                fetched_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
            except Exception as e:
                return f"ERROR:: Failed to fetch transcript: {e}"

        # Join chunks and filter common non-speech tokens
        parts = []
        for d in fetched_transcript:
            text = d.get('text', '')
            if text and text not in ('[Music]', '[Applause]'):
                parts.append(text)
        full_transcript = " ".join(parts).strip()

        if not full_transcript:
            return "ERROR:: Empty transcript"

        print("--- Transcript fetched successfully ---")
        return full_transcript

    except Exception as e:
        print(f"--- Error fetching transcript: {e} ---")
        return f"ERROR:: Unexpected failure: {e}"

# --- 3. Configure the LLM and Bind Tools ---
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("The OPENAI_API_KEY environment variable is not set. Please create a .env file and add it.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_transcript])

# --- 4. Manage Chat History ---
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- 5. Create the Core Conversational Chain ---

def format_messages(input_dict: dict) -> list:
    history = input_dict.get('history', [])
    new_messages = input_dict.get('messages', [])
    return history + new_messages

message_processing_chain = RunnableLambda(format_messages) | llm_with_tools

conversational_chain = RunnableWithMessageHistory(
    message_processing_chain,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="history",
)

# --- 6. Define the API Endpoint ---
@app.post("/chat", response_model=ChatOutput)
async def chat_with_video(chat_input: ChatInput):
    session_id = chat_input.uri
    user_question = chat_input.input
    history = get_session_history(session_id)

    # Check if we've already added a transcript into the history in a prior turn
    has_any_tool_msg = any(isinstance(msg, ToolMessage) for msg in history.messages)

    # Instruct the model to either call the tool or answer using the transcript
    prompt_messages = [
        HumanMessage(content=(
            f"You are a helpful assistant answering questions about a YouTube video.\n"
            f"Video URL: {chat_input.uri}\n\n"
            f"User's question: \"{user_question}\"\n\n"
            "If you do NOT have the transcript in the chat history yet, you MUST call the `get_transcript` tool now with the full URL.\n"
            "If the transcript is already present in the history, answer the question using only that content.\n"
            "Always provide a concise answer and a short justification."
        ))
    ]

    ai_response = conversational_chain.invoke(
        {"messages": prompt_messages},
        config={"configurable": {"session_id": session_id}},
    )

    # If the model requested the transcript and we haven't provided one yet, run the tool and continue the conversation
    if getattr(ai_response, "tool_calls", None) and not has_any_tool_msg:
        print("--- LLM decided to call the get_transcript tool ---")
        tool_messages = []
        for call in ai_response.tool_calls:
            # Be defensive about arg shape
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            url = None
            if isinstance(args, dict):
                url = args.get("video_url") or args.get("url") or args.get("uri") or chat_input.uri
            else:
                url = chat_input.uri

            tool_output = get_transcript.invoke({"video_url": url})

            # If transcript tool failed, propagate a friendly error now rather than letting the model guess
            if isinstance(tool_output, str) and tool_output.startswith("ERROR::"):
                return ChatOutput(
                    answer="I couldn't fetch the transcript for this video.",
                    justification=tool_output.replace("ERROR::", "").strip() or "The transcript is unavailable for this video.",
                )

            tool_messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=(call.get("id") if isinstance(call, dict) else getattr(call, "id", "tool_call")))
            )

        print("--- Invoking LLM again with the transcript ---")
        # Provide the tool results; history remembers the initial human and assistant messages
        ai_response = conversational_chain.invoke(
            {"messages": tool_messages},
            config={"configurable": {"session_id": session_id}},
        )

    # --- 7. Produce a structured final JSON answer ---
    final_llm = llm.with_structured_output(ChatOutput)
    final_prompt = (
        "Based on the conversation so far (including any transcript present in the tool messages), "
        "answer the user's last question in two short fields: `answer` and `justification`.\n\n"
        f"User's last question: \"{user_question}\"\n\n"
        f"Your last content message (if any): \"{getattr(ai_response, 'content', '')}\""
    )

    final_structured_response = final_llm.invoke(final_prompt)
    return final_structured_response

# --- 8. Run the Application ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
