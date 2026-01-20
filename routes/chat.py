"""
Chat Routes Module
Defines API endpoints for the chat functionality.
"""

from flask import Blueprint, request, jsonify
from agents.car_expert import CarExpertAgent
import markdown

# Create Blueprint for chat routes
chat_bp = Blueprint("chat", __name__)

# Initialize the Car Expert Agent (singleton pattern)
car_agent = None


def get_agent():
    """Get or create the Car Expert Agent instance."""
    global car_agent
    if car_agent is None:
        car_agent = CarExpertAgent()
    return car_agent


def render_markdown(text: str) -> str:
    """Convert markdown to HTML."""
    return markdown.markdown(
        text, 
        extensions=['tables', 'fenced_code', 'nl2br']
    )


@chat_bp.route("/ask", methods=["POST"])
def ask_question():
    """
    Handle user questions about cars.
    
    Expects JSON payload: 
    {
        "question": "user's question here",
        "image": "base64 encoded image (optional)"
    }
    """
    try:
        # Validate request content type
        if not request.is_json:
            return jsonify({
                "response": "Invalid request format. Please send JSON data.",
                "status": "error"
            }), 400
        
        # Get data from request body
        data = request.get_json()
        question = data.get("question", "").strip()
        image_data = data.get("image", None)
        
        # Validate - need either question or image
        if not question and not image_data:
            return jsonify({
                "response": "Please provide a question or upload an image.",
                "status": "error"
            }), 400
        
        # Default question for image-only requests
        if image_data and not question:
            question = "What car is this? Tell me about it!"
        
        # Get agent and process the question
        agent = get_agent()
        result = agent.ask(question, image_data)
        
        # Render markdown to HTML
        response_html = render_markdown(result.get("response", ""))
        
        return jsonify({
            "response": response_html,
            "response_raw": result.get("response", ""),
            "image_links": result.get("image_links", []),
            "search_used": result.get("search_used", False),
            "image_analyzed": result.get("image_analyzed", False),
            "status": "success"
        }), 200
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            "response": "An unexpected error occurred. Please try again.",
            "status": "error",
            "error_detail": str(e)
        }), 500


@chat_bp.route("/clear", methods=["POST"])
def clear_chat():
    """Clear conversation history to start a new chat."""
    try:
        agent = get_agent()
        agent.clear_history()
        return jsonify({
            "status": "success",
            "message": "Chat history cleared!"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@chat_bp.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the service is running.
    Also checks if Ollama is accessible.
    """
    try:
        agent = get_agent()
        ollama_status = agent.health_check()
        
        return jsonify({
            "status": "healthy",
            "ollama_connected": ollama_status,
            "message": "AutoMind is running!" if ollama_status else "AutoMind is running but Ollama may not be accessible."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "ollama_connected": False,
            "message": str(e)
        }), 503
