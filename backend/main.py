from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from services.genai import YoutubeProcessor, GeminiProcessor

class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):

    genai_processor = GeminiProcessor(
        model_name="gemini-pro",
        project="dynamo-425102",

    )

    # Calling YouTube Processor class to extract video data and return data to
    # back to the front end

    processor = YoutubeProcessor(genai_processor=genai_processor)

    result = processor.retrieve_youtube_documents(str(request.youtube_link), verbose=True)

    # summary = genai_processor.generate_document_summary(result, verbose=True)

    key_concepts = processor.find_key_concepts(result, group_size=10, verbose=True)

    return {"key_concepts": key_concepts}
