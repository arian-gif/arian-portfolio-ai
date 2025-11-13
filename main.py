from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
from openai import OpenAI
import json
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

load_dotenv()

class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: str # 'user', 'assistant', or 'system'
    content: str

class ChatRequest(BaseModel):
    """The request body for the /api/chat endpoint."""
    messages: List[Message]
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """The response body for the /api/chat endpoint."""
    message: Message
    session_id: str

def submit_contact_request(name: str, email: str, message: str) -> Dict[str, str]:
    """
    Submits a contact request with user's name, email, and message.
    Sends an email notification using Gmail SMTP.
    """
    MY_EMAIL = os.getenv("MY_EMAIL")
    MY_PASSWORD = os.getenv("MY_PASSWORD")
    OTHER_EMAIL = os.getenv("OTHER_EMAIL")

    # Compose email
    subject = f"New Contact Request from {name}"
    body = f"""
    You received a new contact request:

    Name: {name}
    Email: {email}
    Message:
    {message}

    Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = MY_EMAIL
    msg["To"] = OTHER_EMAIL

    # Send email securely
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MY_EMAIL, MY_PASSWORD)
            server.send_message(msg)
            print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return {"status": "error", "message": "There was an issue sending your message. Please try again later."}

    return {
        "status": "success",
        "message": f"Thank you, {name}! Your message has been submitted. Arian will get back to you at {email} soon."
    }

def get_project_details(project_name: str) -> Dict[str, Any]:
    """
    Retrieves detailed information about a specific project from the resume data.
    """
    project_name = project_name.lower()
    
    # Check personal projects
    for project in Resume_Info.get("personal_projects", []):
        if project_name in project["title"].lower():
            return project
    
    # Check work experience
    for job in Resume_Info.get("work_experience", []):
        if project_name in job["company"].lower() or project_name in job["title"].lower():
            return job

    return {"status": "not_found", "message": f"Could not find detailed information for a project named '{project_name}'. Please try another name."}

# Map tool names to actual functions
available_functions = {
    "submit_contact_request": submit_contact_request,
    "get_project_details": get_project_details,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "submit_contact_request",
            "description": "Submits a contact request from a visitor to Arian. Requires the visitor's name, email, and message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the visitor."},
                    "email": {"type": "string", "description": "The email address of the visitor."},
                    "message": {"type": "string", "description": "The message the visitor wants to send to Arian."},
                },
                "required": ["name", "email", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_details",
            "description": "Retrieves the full details (description, technologies, etc.) for a specific personal project or work experience item when asked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "The name of the project or company to get details for, e.g., 'Fantasy Draft' or 'Algoverse'."},
                },
                "required": ["project_name"],
            },
        },
    }
]

app = FastAPI(title="Portfolio AI Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "https://ariankhan.netlify.app",
        "https://ariankhan.ca",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

try:
    with open('resume_info.json', 'r') as file:
        Resume_Info = json.load(file)
except FileNotFoundError:
    print("Error: 'resume_info.json' not found. Please create it using the content below.")
    Resume_Info = {} 

arian_info = """
Hey, this is Arian. I love the gym, music, sports, and a good show to unwind.
I stick to a push pull legs routine, and when I’m not lifting, I’m probably listening to The Weeknd.
I’m big on shows like Brooklyn Nine-Nine and Stranger Things, 
and I’ll even throw on anime like Attack on Titan or Bleach when I’m in the mood for chaos.
I’m a huge fan of sports too, especially football and basketball, go Raptors! 
I’ll even admit, I became a Jays fan the second they made the World Series. Outside of that,
I’ve been learning guitar, trying to keep my Spanish streak alive on Duolingo, 
and playing chess whenever I get the chance. I try to keep life simple, learn a lot,
laugh often, lift heavy, and have fun along the way.
"""

PORTFOLIO_INFO = f"""
Arian Khan is an Engineering student passionate about building intelligent systems that bridge human and machine interaction. 
He combines expertise in deep learning, full-stack development, and embedded systems to create practical, end-to-end AI applications.

His resume is as follows:
{json.dumps(Resume_Info)}

Here's some info on Arian:
{arian_info}

Arian’s mission is to push the boundaries of AI usability — developing systems that think, learn, and communicate seamlessly with humans.
"""


SYSTEM_PROMPT = f"""You are Arian's AI portfolio assistant. You're helpful, friendly, and knowledgeable about Arian's work.

Your role:
1. Answer questions about Arian's projects, skills, and experience
2. Facilitate contact requests when people want to reach out
3. Provide detailed technical information when asked
4. Be enthusiastic about his work but remain professional

Portfolio Information:
{PORTFOLIO_INFO}

Guidelines:
- Be conversational and friendly
- Provide specific details when discussing projects
- When someone wants to contact Arian, ask for their name, email, and message, then use the submit_contact_request tool
- For project details, use get_project_details tool
- Always be honest if you don't know something
- Encourage meaningful connections and collaborations
- Try to crack a few jokes now and then
- Be short and concise like 1-2 sentences or roughly 40–60 words unless detail is explicitly requested

"""
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message
        
        if assistant_message.tool_calls:
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    function_args = {}
                
                function_response = available_functions[function_name](**function_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
            
            final_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            assistant_message = final_response.choices[0].message
        
        session_id_to_return = request.session_id if request.session_id else "default"
        
        return ChatResponse(
            message=Message(
                role="assistant",
                content=assistant_message.content
            ),
            session_id=session_id_to_return
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please check the backend logs.")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Portfolio AI Agent"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Portfolio AI Agent API",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)