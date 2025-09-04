import os, uvicorn
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

# ---------- ENV -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in .env")

# ---------- FASTAPI -------------
app = FastAPI(
    title="YouTube Video Chatbot API",
    description="Chat with a YouTube video",
    version="1.1.0",
)

# ---------- Pydantic Models -----
class ChatInput(BaseModel):
    uri: str = Field(..., example="https://youtu.be/QsYGlZkevEg")
    input: str = Field(..., example="What were the main announcements?")

class ChatOutput(BaseModel):
    answer: str
    justification: str

# ---------- Tool ----------------
@tool
def get_transcript(video_url: str) -> str:
    """Return full transcript text for a YouTube video URL."""
    parsed = urlparse(video_url)
    if "youtube.com" in parsed.netloc:
        vid = parse_qs(parsed.query).get("v", [None])[0]
    elif "youtu.be" in parsed.netloc:
        vid = parsed.path.lstrip("/")
    else:
        raise ValueError("Invalid YouTube URL")

    if not vid:
        raise ValueError("Could not extract video ID")

    try:
        # prefer manual transcript; fallback to auto-generated
        transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
        transcript = transcript_list.find_transcript(["en", "hi"])
        full = " ".join(chunk["text"] for chunk in transcript.fetch())
        return full or "Transcript empty"
    except Exception as e:
        return f"Error fetching transcript: {e}"

# ---------- LLM + tools ---------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tool_llm = llm.bind_tools([get_transcript])

def merge_msgs(d):
    return d["history"] + d["messages"]

chain = RunnableLambda(merge_msgs) | tool_llm

store: dict[str, ChatMessageHistory] = {}
def get_hist(session_id: str):
    return store.setdefault(session_id, ChatMessageHistory())

convo = RunnableWithMessageHistory(
    chain,
    get_hist,
    input_messages_key="messages",
    history_messages_key="history",
)

# ---------- Endpoint ------------
@app.post("/chat", response_model=ChatOutput)
async def chat(chat_input: ChatInput):
    session_id = chat_input.uri
    user_msg = HumanMessage(
        content=(
            "You are a helpful assistant for the YouTube video {url}. "
            "Answer the user's question. If the transcript is not in memory, "
            "call get_transcript first."
        ).format(url=chat_input.uri)
    )
    question_msg = HumanMessage(content=chat_input.input)

    response = convo.invoke(
        {"messages": [user_msg, question_msg]},
        config={"configurable": {"session_id": session_id}},
    )

    # if the model called the tool, supply its result and re-invoke once
    if response.tool_calls:
        tool_outputs = []
        for tc in response.tool_calls:
            out = get_transcript.invoke(tc["args"])
            tool_outputs.append(
                ToolMessage(content=out, tool_call_id=tc["id"])
            )
        response = convo.invoke(
            {"messages": tool_outputs},
            config={"configurable": {"session_id": session_id}},
        )

    # At this point response.content already contains JSON-parsable dict
    if isinstance(response.content, dict):
        return ChatOutput(**response.content)

    # fallback – let model format explicitly
    final = llm.with_structured_output(ChatOutput).invoke(
        f"Respond with JSON {{'answer','justification'}} for: {response.content}"
    )
    return final

# ---------- Main ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
