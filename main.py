from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import tempfile
from faster_whisper import WhisperModel
import logging
import ollama
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
from typing import List, Optional, Set
import json
import re # Import the re module for regular expressions
import string

# Load environment variables
load_dotenv()

# Configuration from environment variables
HOST_IP = os.getenv("HOST_IP", "192.168.1.35")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SSL_KEY_PATH = os.getenv("SSL_KEY_PATH", "key.pem")
SSL_CERT_PATH = os.getenv("SSL_CERT_PATH", "cert.pem")

def extract_json_from_response(text: str) -> Optional[dict]:
    """
    Extracts a JSON object from a string that might contain additional text or markdown.
    It looks for a JSON block enclosed in ```json ... ```. If not found, it attempts
    to parse the entire string as JSON.
    """
    # Regex to find a JSON block within markdown code fences
    json_match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass # Fallback to full text parsing if markdown JSON is malformed

    # If no markdown JSON block, or if it was malformed, try to parse the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Notetaker Backend",
    description="FastAPI backend for AI-powered meeting notes with local transcription",
    version="1.0.0"
)

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="src"), name="static")

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"https://{HOST_IP}:8000",
        "https://localhost:8000",
        "https://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
whisper_model = None
ollama_model_name = OLLAMA_MODEL  # Use environment variable
ollama_client = None
chroma_client = None
embedding_model = None
notes_collection = None
meetings_collection = None

# Pydantic models for request/response
class SummarizeRequest(BaseModel):
    text: str
    
class SummarizeResponse(BaseModel):
    data: dict
    note_id: str
    title: str  # Add generated title

class ChatRequest(BaseModel):
    message: str
    meeting_id: Optional[str] = None  # Optional meeting filter
    
class ChatResponse(BaseModel):
    response: str
    sources: List[dict]

class Meeting(BaseModel):
    meeting_id: str
    title: str
    data: dict
    transcription: str
    created_at: datetime

class UpdateMeetingRequest(BaseModel):
    title: str
    data: dict

class ResummarizeResponse(BaseModel):
    meeting_id: str
    title: str
    data: dict
    
class MeetingListResponse(BaseModel):
    meetings: List[dict]

class NoteChunk(BaseModel):
    chunk_id: str
    note_id: str
    content: str
    timestamp: datetime
    chunk_index: int

@app.on_event("startup")
async def startup_event():
    """
    Initialize all models and databases when the application starts.
    """
    global whisper_model, ollama_client, ollama_model_name, chroma_client, embedding_model, notes_collection, meetings_collection
    try:
        logger.info("Loading Whisper model...")
        # Initialize the Whisper model - 'base' model provides good accuracy with reasonable speed
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully!")
        
        # Initialize Ollama
        logger.info("Initializing Ollama client...")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_client = ollama.Client(host=ollama_host)
        
        # Check if the model is available
        try:
            logger.info("Connecting to Ollama and fetching available models...")
            ollama_models_response = ollama_client.list()
            
            if 'models' not in ollama_models_response:
                raise ValueError("Invalid response from Ollama API, 'models' key not found.")

            ollama_models = [m.get('name') for m in ollama_models_response['models']]
            
            if ollama_model_name not in ollama_models:
                logger.warning(f"Model '{ollama_model_name}' not found. Attempting to pull it...")
                ollama_client.pull(ollama_model_name)
                logger.info(f"Successfully pulled '{ollama_model_name}'.")
            else:
                logger.info(f"Ollama model '{ollama_model_name}' is available.")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.error(f"Please ensure the Ollama server is running and accessible at '{ollama_host}'.")
            raise
            
        logger.info(f"Ollama initialized successfully with model: {ollama_model_name}")
        
        # Initialize ChromaDB (disable telemetry)
        logger.info("Initializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        notes_collection = chroma_client.get_or_create_collection(
            name="notes",
            metadata={"hnsw:space": "cosine"}
        )
        meetings_collection = chroma_client.get_or_create_collection(
            name="meetings",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB initialized successfully!")
        
        # Initialize sentence transformer for embeddings
        logger.info("Loading sentence transformer model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

@app.get("/")
async def root():
    """
    Serve the main HTML file for the frontend.
    """
    return FileResponse("src/index.html", media_type="text/html")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    Returns a simple success message.
    """
    return {
        "message": "AI Notetaker Backend is running!",
        "status": "healthy",
        "whisper_model_loaded": whisper_model is not None,
        "ollama_client_loaded": ollama_client is not None,
        "chroma_db_loaded": chroma_client is not None,
        "embedding_model_loaded": embedding_model is not None,
        "meetings_collection_loaded": meetings_collection is not None
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Main transcription endpoint that accepts an audio file and returns transcribed text.
    
    Args:
        file: UploadFile object containing the audio file to transcribe
        
    Returns:
        JSON response with the transcribed text
        
    Process:
        1. Validate the uploaded file
        2. Save the file temporarily to disk
        3. Use faster-whisper to transcribe the audio
        4. Clean up the temporary file
        5. Return the transcription result
    """
    
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")
    
    # Validate file type (basic check)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check if file has audio extension (basic validation)
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.webm']
    file_extension = os.path.splitext(file.filename.lower())[1]
    if file_extension not in audio_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported formats: {', '.join(audio_extensions)}"
        )
    
    # Create a temporary file to store the uploaded audio
    temp_file = None
    try:
        # Create temporary file with the same extension as the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Read the uploaded file content and write it to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        logger.info(f"Processing audio file: {file.filename} ({len(content)} bytes)")
        
        # Transcribe the audio using faster-whisper
        segments, info = whisper_model.transcribe(
            temp_file_path,
            beam_size=1,  # Faster transcription with beam_size=1
            language=None,  # Auto-detect language
            vad_filter=True,  # Voice Activity Detection to filter out silence
            vad_parameters=dict(min_silence_duration_ms=500)  # Minimum silence duration
        )
        
        # Collect all transcribed text segments
        transcription_text = ""
        for segment in segments:
            transcription_text += segment.text + " "
        
        # Clean up the transcription (remove extra spaces)
        transcription_text = transcription_text.strip()
        
        logger.info(f"Transcription completed. Language detected: {info.language} (confidence: {info.language_probability:.2f})")
        
        # Return the transcription result
        return {
            "transcription": transcription_text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        
    finally:
        # Clean up: remove the temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Summarize transcribed text using a local Ollama model.
    Optimized for thinking models like gpt-oss that output reasoning first.
    """

    if ollama_client is None:
        raise HTTPException(status_code=500, detail="Ollama client not loaded")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for summarization")

    try:
        logger.info(f"Summarizing text ({len(request.text)} characters) with model {ollama_model_name}")

        # Truncate text if it's too long
        max_chars = 100000  # Increased limit - most models can handle this
        #text_to_process = request.text
        text_to_process = preprocess_for_summarization(request.text, aggressive=True)
        if len(text_to_process) > max_chars:
            text_to_process = text_to_process[:max_chars] + "..."
            logger.warning(f"Text truncated from {len(request.text)} to {len(text_to_process)} characters")
        else:
            logger.info(f"Using full text: {len(text_to_process)} characters")

        # Prompt optimized for thinking models - ask for final answer clearly
        prompt = f"""Please analyze this meeting transcription and provide a structured summary.

Text to analyze:
{text_to_process}

Think through the content, then provide your final answer as a JSON object with this exact structure:
{{
    "summary": "Brief overview of the meeting",
    "key_points": ["Important point 1", "Important point 2"],
    "action_items": ["Action item 1", "Action item 2"],
    "categories": ["Category 1", "Category 2"]
}}

Make sure your final JSON response is properly formatted and complete."""

        try:
            logger.info("Generating response with thinking model...")
            
            # For thinking models, remove format='json' and increase token limit
            response = ollama_client.chat(
                model=ollama_model_name,
                messages=[{'role': 'user', 'content': prompt}],
                # Remove format='json' - let the model think freely
                options={
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 40,
                    'num_predict': 2000,  # Increase for thinking tokens + response
                    'stop': [],  # Let it complete fully
                }
            )
            response_text = response.get('message', {}).get('content', '').strip()
            logger.info(f"Full response received: {len(response_text)} characters")
            
            # For debugging - log a portion of the raw response
            if response_text:
                logger.info(f"Response preview: {response_text[:200]}...")
                # Print full response to stdout for debugging
                print("=" * 80)
                print("FULL OLLAMA RESPONSE:")
                print("=" * 80)
                print(response_text)
                print("=" * 80)
            
        except Exception as generation_error:
            logger.error(f"Response generation failed: {generation_error}")
            raise HTTPException(status_code=500, detail="Model generation failed")

        if not response_text:
            raise HTTPException(status_code=500, detail="Model returned empty response")

        # Extract JSON from thinking model response
        # The JSON will typically be at the end after the reasoning
        result = extract_json_from_thinking_response(response_text)
        
        if result is None:
            logger.warning("Could not extract JSON from thinking response, trying fallback parsing...")
            
            # Fallback: look for JSON-like content anywhere in the response
            result = extract_json_from_response(response_text)
            
            if result is None:
                logger.error("Complete JSON extraction failure, creating fallback response")
                
                # Last resort: extract information manually from the response
                summary_text = extract_summary_from_text(response_text)
                
                result = {
                    "summary": summary_text or "Unable to generate summary",
                    "key_points": extract_key_points_from_text(response_text),
                    "action_items": extract_action_items_from_text(response_text),
                    "categories": ["General Discussion"]
                }

        # Validate and clean up the result
        result = validate_and_clean_result(result)

        logger.info("Text summarization completed successfully")

        # Generate unique note ID for this transcription
        note_id = str(uuid.uuid4())

        # Generate meeting title
        summary_for_title = result.get("summary", "")
        title = await generate_meeting_title(request.text, summary_for_title)

        # Save meeting to database
        await save_meeting(note_id, title, result, request.text)

        # Index the transcription for future chat queries
        await index_transcription(request.text, note_id)

        return SummarizeResponse(
            data=result,
            note_id=note_id,
            title=title
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")



def extract_json_from_thinking_response(response_text: str) -> dict:
    """
    Extract JSON from a thinking model response.
    Thinking models output reasoning first, then the final answer, often in markdown code blocks.
    """
    import json
    import re
    
    # Strategy 1: Look for JSON in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    code_matches = re.findall(code_block_pattern, response_text, re.DOTALL)
    
    for match in code_matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Look for JSON at the end of the response
    lines = response_text.strip().split('\n')
    
    # Try to find JSON in the last few lines
    for i in range(len(lines), max(0, len(lines) - 15), -1):
        potential_json = '\n'.join(lines[i-1:])
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', potential_json, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            continue
    
    # Strategy 3: Look for the largest JSON object in the entire response
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    for match in reversed(json_matches):  # Try largest/last matches first
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    print("DEBUG: Failed to extract JSON from response")
    print("Response text:")
    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
    
    return None

def validate_and_clean_result(result: dict) -> dict:
    """Ensure the result has all required fields with proper types"""
    if not isinstance(result, dict):
        result = {}
    
    # Set defaults for missing fields
    result.setdefault("summary", "Summary not available")
    result.setdefault("key_points", [])
    result.setdefault("action_items", [])
    result.setdefault("categories", ["General"])
    
    # Ensure lists are actually lists
    for key in ["key_points", "action_items", "categories"]:
        if not isinstance(result[key], list):
            if isinstance(result[key], str):
                result[key] = [result[key]]
            else:
                result[key] = []
    
    # Ensure summary is a string
    if not isinstance(result["summary"], str):
        result["summary"] = str(result.get("summary", "Summary not available"))
    
    return result


def extract_summary_from_text(text: str) -> str:
    """Extract a summary from unstructured text response"""
    lines = text.split('\n')
    
    # Look for lines that might contain summary
    summary_indicators = ['summary', 'overview', 'in summary', 'to summarize']
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for indicator in summary_indicators:
            if indicator in line_lower:
                # Take this line and a few following lines
                summary_lines = lines[i:i+3]
                summary = ' '.join(summary_lines).strip()
                if len(summary) > 20:  # Reasonable length
                    return summary[:500]  # Limit length
    
    # Fallback: take first substantial paragraph
    for line in lines:
        if len(line.strip()) > 50:
            return line.strip()[:500]
    
    return text[:200] + "..." if len(text) > 200 else text


def extract_key_points_from_text(text: str) -> list:
    """Extract key points from unstructured text"""
    lines = text.split('\n')
    points = []
    
    # Look for bullet points, numbered lists, or lines with key indicators
    key_indicators = ['key point', 'important', 'main point', '•', '-', '*']
    
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # Reasonable length
            line_lower = line.lower()
            
            # Check if line starts with bullet/number or contains key indicators
            if (line.startswith(('•', '-', '*', '1.', '2.', '3.')) or 
                any(indicator in line_lower for indicator in key_indicators)):
                points.append(line[:200])  # Limit length
                
            if len(points) >= 5:  # Reasonable limit
                break
    
    return points if points else ["No specific key points identified"]



# Text chunking and indexing functions
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better semantic search.
    
    Args:
        text: The text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        if end >= len(words):
            break
            
        start = end - overlap
    
    return chunks

async def index_transcription(transcription: str, note_id: str):
    """
    Index a transcription for semantic search by chunking and creating embeddings.
    
    Args:
        transcription: The transcribed text
        note_id: Unique identifier for this note
    """
    if not transcription or not transcription.strip():
        logger.warning(f"Empty transcription for note {note_id}, skipping indexing")
        return
    
    try:
        logger.info(f"Indexing transcription for note {note_id}")
        
        # Chunk the transcription
        chunks = chunk_text(transcription)
        
        # Get current timestamp
        timestamp = datetime.now()
        
        # Create embeddings for each chunk
        embeddings = embedding_model.encode(chunks)
        
        # Prepare data for ChromaDB
        chunk_ids = []
        metadatas = []
        documents = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{note_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "note_id": note_id,
                "chunk_index": i,
                "timestamp": timestamp.isoformat(),
                "word_count": len(chunk.split())
            })
        
        # Add to ChromaDB
        notes_collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully indexed {len(chunks)} chunks for note {note_id}")
        
    except Exception as e:
        logger.error(f"Failed to index transcription for note {note_id}: {e}")
        # Don't raise the exception as indexing failure shouldn't break summarization

@app.post("/chat", response_model=ChatResponse)
async def chat_with_notes(request: ChatRequest):
    """
    Chat endpoint that performs semantic search over stored notes and generates responses.
    
    Args:
        request: ChatRequest containing the user's message
        
    Returns:
        JSON response with AI-generated answer and source information
    """
    
    if ollama_client is None:
        raise HTTPException(status_code=500, detail="Ollama client not loaded")
    
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    
    if notes_collection is None or meetings_collection is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="No message provided")
    
    try:
        logger.info(f"Processing chat query: {request.message[:100]}...")
        
        # Create embedding for the user's question
        query_embedding = embedding_model.encode([request.message]).tolist()
        
        # --- Search both collections ---
        
        # 1. Search transcription chunks (notes_collection)
        note_where_filter = {"note_id": request.meeting_id} if request.meeting_id else None
        note_results = notes_collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=["documents", "metadatas", "distances"],
            where=note_where_filter
        )
        
        # 2. Search meeting summaries (meetings_collection)
        meeting_where_filter = {"meeting_id": request.meeting_id} if request.meeting_id else None
        meeting_results = meetings_collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"],
            where=meeting_where_filter
        )
        
        # --- Combine and rank results ---
        
        all_results = []
        
        # Process transcription chunk results
        if note_results["documents"] and note_results["documents"][0]:
            for doc, meta, dist in zip(note_results["documents"][0], note_results["metadatas"][0], note_results["distances"][0]):
                all_results.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "type": "transcription_chunk"
                })
                
        # Process meeting summary results
        if meeting_results["documents"] and meeting_results["documents"][0]:
            for doc, meta, dist in zip(meeting_results["documents"][0], meeting_results["metadatas"][0], meeting_results["distances"][0]):
                meeting_data = json.loads(doc)
                # Access summary from the 'data' sub-dictionary
                summary_content = meeting_data.get('data', {}).get('summary', 'No summary available')
                all_results.append({
                    "content": f"Title: {meeting_data['title']}\nSummary: {summary_content}",
                    "metadata": meta,
                    "distance": dist,
                    "type": "meeting_summary"
                })

        if not all_results:
            return ChatResponse(
                response="I don't have any notes to reference yet. Please transcribe some audio first so I can help answer questions about your recordings.",
                sources=[]
            )
            
        # Sort all results by distance (lower is better)
        all_results.sort(key=lambda x: x["distance"])
        
        # --- Construct context and sources from the top results ---
        
        top_results = all_results[:5] # Use top 5 combined results
        
        context_parts = []
        sources = []
        
        for result in top_results:
            relevance_score = round(1 - result["distance"], 3)
            if relevance_score < 0.2: continue # Filter out very low-relevance results

            if result["type"] == "transcription_chunk":
                timestamp_str = datetime.fromisoformat(result["metadata"]["timestamp"]).strftime("%Y-%m-%d at %H:%M")
                context_parts.append(f"[From original transcription recorded on {timestamp_str}]:\n{result['content']}")
                sources.append({
                    "note_id": result["metadata"]["note_id"],
                    "timestamp": result["metadata"]["timestamp"],
                    "relevance_score": relevance_score,
                    "preview": result["content"][:150] + "...",
                    "type": "Transcription"
                })
            
            elif result["type"] == "meeting_summary":
                timestamp_str = datetime.fromisoformat(result["metadata"]["created_at"]).strftime("%Y-%m-%d")
                context_parts.append(f"[From your saved summary for meeting on {timestamp_str}]:\n{result['content']}")
                sources.append({
                    "note_id": result["metadata"]["meeting_id"],
                    "timestamp": result["metadata"]["created_at"],
                    "relevance_score": relevance_score,
                    "preview": result["content"][:150] + "...",
                    "type": "Summary"
                })

        if not context_parts:
            return ChatResponse(
                response="I couldn't find any relevant information in your notes to answer that question. Try asking about topics you've discussed in your recorded conversations.",
                sources=[]
            )
            
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt for Gemini
        prompt = f"""You are an AI assistant. Based on the following context from a user's notes, which includes raw transcriptions and their edited summaries, answer the user's question.

Context from notes:
---
{context}
---

User's question: {request.message}

Provide a helpful, conversational response based on the provided context.
"""

        # Generate response using Ollama
        response = ollama_client.chat(
            model=ollama_model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        response_text = response['message']['content']

        if not response_text:
            raise HTTPException(status_code=500, detail="Failed to generate response from Ollama")
        
        logger.info("Chat response generated successfully")
        
        return ChatResponse(
            response=response_text,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# Meeting management functions
async def generate_meeting_title(transcription: str, summary: str) -> str:
    """
    Generate a concise meeting title based on transcription and summary.
    
    Args:
        transcription: The full transcription text
        summary: The generated summary
        
    Returns:
        Generated meeting title
    """
    try:
        # Create a prompt to generate a meeting title
        prompt = f"""Generate a concise, descriptive title (maximum 6 words) for a meeting based on the following summary and transcription excerpt. Return ONLY the title itself, with no extra text or quotes.

Summary: {summary[:200]}...
Transcription excerpt: {transcription[:300]}...
"""

        response = ollama_client.chat(
            model=ollama_model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        if response and response['message']['content']:
            title = response['message']['content'].strip().replace('"', '').replace("'", "")
            # Ensure title is not too long
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        else:
            # Fallback title based on timestamp
            return f"Meeting {datetime.now().strftime('%m/%d %H:%M')}"
            
    except Exception as e:
        logger.error(f"Error generating meeting title: {e}")
        return f"Meeting {datetime.now().strftime('%m/%d %H:%M')}"

async def save_meeting(meeting_id: str, title: str, data: dict, transcription: str):
    """
    Save meeting data to ChromaDB for retrieval.
    
    Args:
        meeting_id: Unique meeting identifier
        title: Generated meeting title
        data: The flexible JSON data from the model
        transcription: Full transcription text
    """
    try:
        logger.info(f"Saving meeting: {title}")
        
        # Create meeting data
        meeting_data = {
            "meeting_id": meeting_id,
            "title": title,
            "data": data,
            "transcription": transcription,
            "created_at": datetime.now().isoformat()
        }
        
        # Create embedding for the meeting (for search purposes)
        # We'll convert the data dict to a string for embedding
        meeting_text_from_data = " ".join(f"{k}: {v}" for k, v in data.items())
        meeting_text = f"{title} {meeting_text_from_data}"
        embedding = embedding_model.encode([meeting_text])
        
        # Extract categories for metadata, if they exist
        categories = data.get("categories", [])
        if not isinstance(categories, list):
            categories = [str(categories)]

        # Save to ChromaDB
        meetings_collection.add(
            ids=[meeting_id],
            embeddings=embedding.tolist(),
            documents=[json.dumps(meeting_data)],
            metadatas=[{
                "meeting_id": meeting_id,
                "title": title,
                "created_at": meeting_data["created_at"],
                "categories": ",".join(categories)
            }]
        )
        
        logger.info(f"Successfully saved meeting: {title}")
        
    except Exception as e:
        logger.error(f"Failed to save meeting {meeting_id}: {e}")

@app.get("/meetings", response_model=MeetingListResponse)
async def get_meetings():
    """
    Retrieve all meetings sorted by creation date (newest first).
    
    Returns:
        List of meetings with metadata
    """
    try:
        logger.info("Retrieving meeting history")
        
        if meetings_collection is None:
            raise HTTPException(status_code=500, detail="Meetings database not initialized")
        
        # Get all meetings
        results = meetings_collection.get(
            include=["documents", "metadatas"]
        )
        
        if not results["documents"]:
            return MeetingListResponse(meetings=[])
        
        meetings = []
        for doc, metadata in zip(results["documents"], results["metadatas"]):
            try:
                meeting_data = json.loads(doc)
                # Adapt to the new flexible data structure
                summary_preview = ""
                if "data" in meeting_data and "summary" in meeting_data["data"]:
                    summary_preview = meeting_data["data"]["summary"]
                elif "summary" in meeting_data: # For backwards compatibility
                    summary_preview = meeting_data["summary"]

                categories_preview = []
                if "data" in meeting_data and "categories" in meeting_data["data"]:
                    categories_preview = meeting_data["data"]["categories"]
                elif "categories" in meeting_data: # For backwards compatibility
                    categories_preview = meeting_data["categories"]


                meetings.append({
                    "meeting_id": meeting_data["meeting_id"],
                    "title": meeting_data["title"],
                    "summary": summary_preview[:150] + "..." if len(summary_preview) > 150 else summary_preview,
                    "created_at": meeting_data["created_at"],
                    "categories": categories_preview
                })
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing meeting data: {e}")
                continue
        
        # Sort by creation date (newest first)
        meetings.sort(key=lambda x: x["created_at"], reverse=True)
        
        logger.info(f"Retrieved {len(meetings)} meetings")
        return MeetingListResponse(meetings=meetings)
        
    except Exception as e:
        logger.error(f"Error retrieving meetings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve meetings: {str(e)}")

@app.put("/meetings/{meeting_id}")
async def update_meeting(meeting_id: str, request: UpdateMeetingRequest):
    """
    Update an existing meeting's details.
    
    Args:
        meeting_id: The identifier of the meeting to update.
        request: The new data for the meeting.
        
    Returns:
        A confirmation message.
    """
    try:
        logger.info(f"Updating meeting: {meeting_id}")
        
        if meetings_collection is None:
            raise HTTPException(status_code=500, detail="Meetings database not initialized")
        
        # Retrieve the existing meeting to get the transcription
        existing_meeting_result = meetings_collection.get(ids=[meeting_id], include=["documents"])
        
        if not existing_meeting_result["documents"]:
            raise HTTPException(status_code=404, detail="Meeting not found")
            
        existing_meeting_data = json.loads(existing_meeting_result["documents"][0])
        
        # Create the updated meeting data
        updated_meeting_data = {
            "meeting_id": meeting_id,
            "title": request.title,
            "data": request.data,
            "transcription": existing_meeting_data["transcription"],  # Keep original transcription
            "created_at": existing_meeting_data["created_at"]  # Keep original creation date
        }
        
        # Re-create the embedding for the updated content
        meeting_text_from_data = " ".join(f"{k}: {v}" for k, v in request.data.items())
        meeting_text = f"{request.title} {meeting_text_from_data}"
        embedding = embedding_model.encode([meeting_text])

        # Extract categories for metadata, if they exist
        categories = request.data.get("categories", [])
        if not isinstance(categories, list):
            categories = [str(categories)]
        
        # Upsert the updated meeting data into ChromaDB
        meetings_collection.upsert(
            ids=[meeting_id],
            embeddings=embedding.tolist(),
            documents=[json.dumps(updated_meeting_data)],
            metadatas=[{
                "meeting_id": meeting_id,
                "title": request.title,
                "created_at": updated_meeting_data["created_at"],
                "categories": ",".join(categories)
            }]
        )
        
        logger.info(f"Successfully updated meeting: {meeting_id}")
        
        return {"message": "Meeting updated successfully", "meeting_id": meeting_id}
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing existing meeting data for update: {e}")
        raise HTTPException(status_code=500, detail="Invalid existing meeting data")
    except Exception as e:
        logger.error(f"Error updating meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update meeting: {str(e)}")

@app.post("/meetings/{meeting_id}/resummarize", response_model=ResummarizeResponse)
async def resummarize_meeting(meeting_id: str):
    """
    Reprocess the transcription of an existing meeting to generate a new summary.
    Optimized for thinking models like gpt-oss that output reasoning first.
    """
    if ollama_client is None:
        raise HTTPException(status_code=500, detail="Ollama client not loaded")
    if meetings_collection is None:
        raise HTTPException(status_code=500, detail="Meetings database not initialized")
    
    # 1. Retrieve the existing meeting
    try:
        existing_meeting_result = meetings_collection.get(ids=[meeting_id], include=["documents"])
        if not existing_meeting_result["documents"]:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        existing_meeting_data = json.loads(existing_meeting_result["documents"][0])
        transcription = existing_meeting_data.get("transcription")
    
        if not transcription:
            raise HTTPException(status_code=400, detail="Meeting has no transcription to reprocess.")
            
    except Exception as e:
        logger.error(f"Error retrieving meeting {meeting_id} for resummarization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve meeting data: {str(e)}")
    
    # 2. Generate new summary from transcription (using thinking model approach)
    try:
        logger.info(f"Resummarizing transcription for meeting {meeting_id} ({len(transcription)} characters)")
        
        # Truncate text if it's too long for thinking model
        max_chars = 100000  # Increased limit - most models can handle this
        #text_to_process = transcription
        text_to_process = preprocess_for_summarization(transcription, aggressive=True)
        if len(text_to_process) > max_chars:
            text_to_process = text_to_process[:max_chars] + "..."
            logger.warning(f"Transcription truncated from {len(transcription)} to {len(text_to_process)} characters")
        else:
            logger.info(f"Using full transcription: {len(text_to_process)} characters")

        # Prompt optimized for thinking models
        prompt = f"""Please analyze this meeting transcription and provide a structured summary.

Text to analyze:
{text_to_process}

Think through the content, then provide your final answer as a JSON object with this exact structure:
{{
    "summary": "Brief overview of the meeting",
    "key_points": ["Important point 1", "Important point 2"],
    "action_items": ["Action item 1", "Action item 2"],
    "categories": ["Category 1", "Category 2"]
}}

Make sure your final JSON response is properly formatted and complete."""

        try:
            logger.info("Generating new summary with thinking model...")
            
            # Remove format='json' for thinking models and increase token limit
            response = ollama_client.chat(
                model=ollama_model_name,
                messages=[{'role': 'user', 'content': prompt}],
                # Remove format='json' - let the model think freely
                options={
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 40,
                    'num_predict': 8192,  # Increase for thinking tokens + response
                    'stop': [],  # Let it complete fully
                }
            )
            
            response_text = response.get('message', {}).get('content', '').strip()
            logger.info(f"Resummarize response received: {len(response_text)} characters")
            
            if response_text:
                logger.info(f"Response preview: {response_text[:200]}...")
                # Print full response to stdout for debugging
                print("=" * 80)
                print("FULL OLLAMA RESPONSE:")
                print("=" * 80)
                print(response_text)
                print("=" * 80)
            
        except Exception as generation_error:
            logger.error(f"Response generation failed during resummarization: {generation_error}")
            raise HTTPException(status_code=500, detail="Model generation failed during resummarization")

        if not response_text:
            raise HTTPException(status_code=500, detail="Model returned empty response during resummarization")

        # Extract JSON from thinking model response
        new_data = extract_json_from_thinking_response(response_text)
        
        if new_data is None:
            logger.warning("Could not extract JSON from thinking response, trying fallback parsing...")
            
            # Fallback: look for JSON-like content anywhere in the response
            new_data = extract_json_from_response(response_text)
            
            if new_data is None:
                logger.error("Complete JSON extraction failure during resummarization, creating fallback response")
                
                # Last resort: extract information manually from the response
                summary_text = extract_summary_from_text(response_text)
                
                new_data = {
                    "summary": summary_text or "Unable to generate summary",
                    "key_points": extract_key_points_from_text(response_text),
                    "action_items": extract_action_items_from_text(response_text),
                    "categories": ["General Discussion"]
                }

        # Validate and clean up the result
        new_data = validate_and_clean_result(new_data)
        
        logger.info("Resummarization completed successfully")
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error during resummarization for meeting {meeting_id}: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Resummarization failed: {str(e)}")
    
    # 3. Generate new title
    try:
        summary_for_title = new_data.get("summary", "")
        new_title = await generate_meeting_title(transcription, summary_for_title)
    except Exception as e:
        logger.warning(f"Failed to generate new title, using fallback: {e}")
        new_title = f"Meeting {meeting_id[:8]} (Resummarized)"
    
    # 4. Update the meeting in the database
    try:
        updated_meeting_data = {
            "meeting_id": meeting_id,
            "title": new_title,
            "data": new_data,
            "transcription": transcription,
            "created_at": existing_meeting_data.get("created_at", "")
        }
        
        meeting_text_from_data = " ".join(f"{k}: {v}" for k, v in new_data.items())
        meeting_text = f"{new_title} {meeting_text_from_data}"
        embedding = embedding_model.encode([meeting_text])
        
        categories = new_data.get("categories", [])
        if not isinstance(categories, list):
            categories = [str(categories)]
        
        meetings_collection.upsert(
            ids=[meeting_id],
            embeddings=embedding.tolist(),
            documents=[json.dumps(updated_meeting_data)],
            metadatas=[{
                "meeting_id": meeting_id,
                "title": new_title,
                "created_at": updated_meeting_data["created_at"],
                "categories": ",".join(categories)
            }]
        )
        
        logger.info(f"Successfully resummarized and updated meeting: {meeting_id}")
        
        return ResummarizeResponse(
            meeting_id=meeting_id,
            title=new_title,
            data=new_data
        )
        
    except Exception as e:
        logger.error(f"Error updating database during resummarization for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update meeting in database: {str(e)}")


# Helper function to extract JSON from thinking model response (if not already defined)
def extract_json_from_thinking_response(response_text: str) -> dict:
    """
    Extract JSON from a thinking model response.
    Thinking models output reasoning first, then the final answer, often in markdown code blocks.
    """
    import json
    import re
    
    # Strategy 1: Look for JSON in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    code_matches = re.findall(code_block_pattern, response_text, re.DOTALL)
    
    for match in code_matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Look for JSON at the end of the response
    lines = response_text.strip().split('\n')
    
    # Try to find JSON in the last few lines
    for i in range(len(lines), max(0, len(lines) - 15), -1):
        potential_json = '\n'.join(lines[i-1:])
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', potential_json, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            continue
    
    # Strategy 3: Look for the largest JSON object in the entire response
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    for match in reversed(json_matches):  # Try largest/last matches first
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    print("DEBUG: Failed to extract JSON from response")
    print("Response text:")
    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
    
    return None


def validate_and_clean_result(result: dict) -> dict:
    """Ensure the result has all required fields with proper types"""
    if not isinstance(result, dict):
        result = {}
    
    # Set defaults for missing fields
    result.setdefault("summary", "Summary not available")
    result.setdefault("key_points", [])
    result.setdefault("action_items", [])
    result.setdefault("categories", ["General"])
    
    # Ensure lists are actually lists
    for key in ["key_points", "action_items", "categories"]:
        if not isinstance(result[key], list):
            if isinstance(result[key], str):
                result[key] = [result[key]]
            else:
                result[key] = []
    
    # Ensure summary is a string
    if not isinstance(result["summary"], str):
        result["summary"] = str(result.get("summary", "Summary not available"))
    
    return result



def extract_summary_from_text(text: str) -> str:
    """Extract a summary from unstructured text response"""
    lines = text.split('\n')
    
    # Look for lines that might contain summary
    summary_indicators = ['summary', 'overview', 'in summary', 'to summarize']
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for indicator in summary_indicators:
            if indicator in line_lower:
                # Take this line and a few following lines
                summary_lines = lines[i:i+3]
                summary = ' '.join(summary_lines).strip()
                if len(summary) > 20:  # Reasonable length
                    return summary[:500]  # Limit length
    
    # Fallback: take first substantial paragraph
    for line in lines:
        if len(line.strip()) > 50:
            return line.strip()[:500]
    
    return text[:200] + "..." if len(text) > 200 else text


def extract_key_points_from_text(text: str) -> list:
    """Extract key points from unstructured text"""
    lines = text.split('\n')
    points = []
    
    # Look for bullet points, numbered lists, or lines with key indicators
    key_indicators = ['key point', 'important', 'main point', '•', '-', '*']
    
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # Reasonable length
            line_lower = line.lower()
            
            # Check if line starts with bullet/number or contains key indicators
            if (line.startswith(('•', '-', '*', '1.', '2.', '3.')) or 
                any(indicator in line_lower for indicator in key_indicators)):
                points.append(line[:200])  # Limit length
                
            if len(points) >= 5:  # Reasonable limit
                break
    
    return points if points else ["No specific key points identified"]


def extract_action_items_from_text(text: str) -> list:
    """Extract action items from unstructured text"""
    lines = text.split('\n')
    actions = []
    
    action_indicators = ['action', 'todo', 'task', 'follow up', 'next step', 'assign']
    
    for line in lines:
        line = line.strip()
        if len(line) > 10:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in action_indicators):
                actions.append(line[:200])
                
            if len(actions) >= 3:
                break
    
    return actions  # Return empty list if none found

class ConversationPreprocessor:
    """
    Preprocesses conversation transcripts to reduce token count while preserving meaning.
    Removes filler words, repetitions, and noise while maintaining important context.
    """
    
    def __init__(self):
        # Common filler words/phrases in conversations
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'eh', 'mm', 'hmm', 'mhm', 'mmhm', 'erm',
            'like', 'you know', 'i mean', 'sort of', 'kind of', 'basically', 
            'actually', 'literally', 'obviously', 'definitely', 'totally',
            'really', 'very', 'quite', 'pretty much', 'i guess', 'i think',
            'well', 'so', 'anyway', 'alright', 'okay', 'ok', 'right',
            'just', 'maybe', 'perhaps', 'probably', 'honestly', 'frankly'
        }
        
        # Words that often indicate redundant speech patterns
        self.redundant_phrases = [
            r'\b(you know what i mean)\b',
            r'\b(if you will)\b',
            r'\b(so to speak)\b',
            r'\b(as i was saying)\b',
            r'\b(going back to)\b',
            r'\b(what i\'m trying to say is)\b',
            r'\b(the thing is)\b',
            r'\b(at the end of the day)\b'
        ]
        
        # Important business/meeting terms to preserve
        self.important_keywords = {
            'action', 'deadline', 'budget', 'cost', 'revenue', 'profit', 'loss',
            'decision', 'approve', 'reject', 'schedule', 'timeline', 'milestone',
            'responsibility', 'assign', 'complete', 'follow up', 'next steps',
            'priority', 'urgent', 'critical', 'issue', 'problem', 'solution',
            'meeting', 'discussion', 'presentation', 'review', 'analysis'
        }

    def preprocess_conversation(self, text: str, aggressive: bool = False) -> str:
        """
        Main preprocessing function that applies all cleaning steps.
        
        Args:
            text: Raw conversation text
            aggressive: If True, applies more aggressive cleaning (higher compression)
            
        Returns:
            Cleaned and compressed text
        """
        original_length = len(text)
        
        # Step 1: Basic cleaning
        text = self._basic_cleaning(text)
        
        # Step 2: Remove repetitions and stuttering
        text = self._remove_repetitions(text)
        
        # Step 3: Remove filler words (contextually)
        text = self._remove_fillers(text, aggressive)
        
        # Step 4: Remove redundant phrases
        text = self._remove_redundant_phrases(text)
        
        # Step 5: Compress repeated information
        text = self._compress_repetitive_content(text)
        
        # Step 6: Clean up spacing and formatting
        text = self._final_cleanup(text)
        
        compression_ratio = (1 - len(text) / original_length) * 100
        logging.info(f"Text compressed by {compression_ratio:.1f}% ({original_length} -> {len(text)} chars)")
        
        return text

    def _basic_cleaning(self, text: str) -> str:
        """Remove basic noise and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common transcription artifacts
        text = re.sub(r'\[inaudible\]|\[unclear\]|\[crosstalk\]', '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'[?]{2,}', '?', text)    # Multiple question marks
        text = re.sub(r'[!]{2,}', '!', text)    # Multiple exclamation marks
        
        # Remove timestamp markers if present
        text = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]', '', text)
        text = re.sub(r'\(\d{1,2}:\d{2}:\d{2}\)', '', text)
        
        return text.strip()

    def _remove_repetitions(self, text: str) -> str:
        """Remove stuttering and immediate repetitions"""
        # Remove word-level stuttering (e.g., "I I I think" -> "I think")
        text = re.sub(r'\b(\w+)\s+\1\s+\1\b', r'\1', text)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        
        # Remove phrase repetitions (up to 3 words)
        text = re.sub(r'\b(\w+\s+\w+\s+\w+)\s+\1\b', r'\1', text)
        text = re.sub(r'\b(\w+\s+\w+)\s+\1\b', r'\1', text)
        
        return text

    def _remove_fillers(self, text: str, aggressive: bool) -> str:
        """Remove filler words while preserving context"""
        words = text.split()
        cleaned_words = []
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip(string.punctuation)
            
            # Always preserve important business terms
            if word_lower in self.important_keywords:
                cleaned_words.append(word)
                continue
            
            # Check if it's a filler word
            if word_lower in self.filler_words:
                # In aggressive mode, remove most fillers
                if aggressive:
                    # Keep some fillers for readability (every 3rd one)
                    if len(cleaned_words) > 0 and (i % 3 == 0):
                        cleaned_words.append(word)
                else:
                    # In normal mode, keep fillers that provide structure
                    if word_lower in {'well', 'so', 'okay', 'alright'}:
                        # Keep if it starts a sentence or follows a pause
                        if i == 0 or words[i-1].endswith(('.', '!', '?')):
                            cleaned_words.append(word)
                    # Remove pure fillers
                    elif word_lower not in {'um', 'uh', 'er', 'ah', 'like'}:
                        cleaned_words.append(word)
            else:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)

    def _remove_redundant_phrases(self, text: str) -> str:
        """Remove common redundant phrases"""
        for pattern in self.redundant_phrases:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text

    def _compress_repetitive_content(self, text: str) -> str:
        """Identify and compress repetitive content patterns"""
        sentences = re.split(r'[.!?]+', text)
        
        # Remove very similar sentences (simple similarity check)
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
                
            # Check if this sentence is very similar to a recent one
            is_similar = False
            words_set = set(sentence.lower().split())
            
            for recent_sentence in unique_sentences[-3:]:  # Check last 3 sentences
                recent_words_set = set(recent_sentence.lower().split())
                
                # If 70%+ words overlap, consider it repetitive
                if len(words_set & recent_words_set) / max(len(words_set), 1) > 0.7:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences) + '.'

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and formatting"""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])\s+', r'\1 ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove empty sentences
        text = re.sub(r'[.!?]\s*[.!?]', '.', text)
        
        return text.strip()

    def get_preprocessing_stats(self, original: str, processed: str) -> dict:
        """Get detailed statistics about the preprocessing"""
        return {
            'original_length': len(original),
            'processed_length': len(processed),
            'compression_ratio': (1 - len(processed) / len(original)) * 100,
            'original_words': len(original.split()),
            'processed_words': len(processed.split()),
            'words_removed': len(original.split()) - len(processed.split())
        }


# Example usage and integration function
def preprocess_for_summarization(text: str, aggressive: bool = False) -> str:
    """
    Convenience function to preprocess conversation text before LLM summarization.
    
    Args:
        text: Raw conversation transcript
        aggressive: Apply more aggressive compression (use for very long texts)
    
    Returns:
        Preprocessed text ready for LLM summarization
    """
    preprocessor = ConversationPreprocessor()
    return preprocessor.preprocess_conversation(text, aggressive=aggressive)


# Additional utility for chunking if text is still too long
def smart_chunk_conversation(text: str, max_chars: int = 12000) -> List[str]:
    """
    Intelligently chunk conversation text at natural boundaries.
    
    Args:
        text: Preprocessed conversation text
        max_chars: Maximum characters per chunk
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = re.split(r'[.!?]+', text)
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip() + '.'
        
        # If adding this sentence would exceed limit
        if len(current_chunk) + len(sentence) > max_chars:
            if current_chunk:  # Save current chunk if not empty
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:  # Single sentence too long, force split
                chunks.append(sentence[:max_chars])
                current_chunk = sentence[max_chars:]
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# Example integration with your existing function
def preprocess_text_for_llm(text: str, target_length: int = 12000) -> str:
    """
    Complete preprocessing pipeline for LLM input preparation.
    
    Args:
        text: Raw conversation text
        target_length: Target character length (will use aggressive mode if needed)
    
    Returns:
        Preprocessed text ready for LLM
    """
    # First pass: normal preprocessing
    processed = preprocess_for_summarization(text, aggressive=False)
    
    # If still too long, apply aggressive preprocessing
    if len(processed) > target_length:
        processed = preprocess_for_summarization(text, aggressive=True)
    
    # If still too long, take the most important parts (beginning and end)
    if len(processed) > target_length:
        mid_point = target_length // 2
        beginning = processed[:mid_point]
        end = processed[-mid_point:]
        processed = beginning + "\n[... middle content truncated ...]\n" + end
    
    return processed
@app.delete("/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    """
    Delete a meeting and all its associated data.
    
    Args:
        meeting_id: The identifier of the meeting to delete.
        
    Returns:
        A confirmation message.
    """
    try:
        logger.info(f"Deleting meeting: {meeting_id}")
        
        if meetings_collection is None or notes_collection is None:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        # Delete from meetings collection
        meetings_collection.delete(ids=[meeting_id])
        
        # Delete all associated chunks from notes collection
        notes_collection.delete(where={"note_id": meeting_id})
        
        logger.info(f"Successfully deleted meeting {meeting_id} and its associated notes.")
        
        return {"message": "Meeting deleted successfully", "meeting_id": meeting_id}
        
    except Exception as e:
        logger.error(f"Error deleting meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete meeting: {str(e)}")

@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """
    Retrieve a specific meeting by ID.
    
    Args:
        meeting_id: The meeting identifier
        
    Returns:
        Complete meeting data
    """
    try:
        logger.info(f"Retrieving meeting: {meeting_id}")
        
        if meetings_collection is None:
            raise HTTPException(status_code=500, detail="Meetings database not initialized")
        
        # Get specific meeting
        results = meetings_collection.get(
            ids=[meeting_id],
            include=["documents", "metadatas"]
        )
        
        if not results["documents"] or not results["documents"][0]:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        meeting_data = json.loads(results["documents"][0])
        logger.info(f"Retrieved meeting: {meeting_data['title']}")
        
        return meeting_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing meeting data: {e}")
        raise HTTPException(status_code=500, detail="Invalid meeting data")
    except Exception as e:
        logger.error(f"Error retrieving meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve meeting: {str(e)}")

if __name__ == "__main__":
    # Run the server if this script is executed directly
    # To run for network access: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload on code changes during development
        log_level="info",
        ssl_keyfile="../key.pem",
        ssl_certfile="../cert.pem"
    )
