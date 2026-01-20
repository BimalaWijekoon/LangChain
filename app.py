"""
AutoMind: Real-Time Car Enthusiast Advisor
Main Flask Application Entry Point

This application provides an AI-powered chat interface for car enthusiasts.
It uses LangChain with Ollama (llama3) and DuckDuckGo search to provide
real-time, accurate information about vehicles.

Author: AutoMind Team
"""

from flask import Flask, render_template
from flask_cors import CORS
from config.settings import get_config
from routes.chat import chat_bp


def create_app():
    """
    Application Factory Pattern.
    Creates and configures the Flask application.
    
    Returns:
        Flask: Configured Flask application instance.
    """
    # Initialize Flask app
    app = Flask(__name__)
    
    # Load configuration
    config = get_config()
    app.config.from_object(config)
    
    # Enable CORS for frontend requests
    CORS(app)
    
    # Register blueprints (routes)
    app.register_blueprint(chat_bp)
    
    # Root route - serves the chat interface
    @app.route("/")
    def index():
        """Serve the main chat interface."""
        return render_template("index.html")
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return {"error": "Resource not found", "status": 404}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return {"error": "Internal server error", "status": 500}, 500
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🚗 AutoMind: Real-Time Car Enthusiast Advisor 🚗     ║
    ║                                                           ║
    ║     ⚡ LangChain ReAct Agent                              ║
    ║     🔍 DuckDuckGo Search Tool                             ║
    ║     🧠 Conversation Memory                                ║
    ║     📸 Gemini Vision (Image Analysis)                     ║
    ║                                                           ║
    ║     Access the app at: http://localhost:5000              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Run the Flask development server
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
