"""
Search Tools for AutoMind
=========================

This module contains web search, Wikipedia, YouTube and Image tools for gathering
real-time car information.

NLP/LangChain Concepts:
- Tool Definition: Using @tool decorator for LangChain integration
- Web Search: DuckDuckGo for real-time data
- Image Search: DuckDuckGo images for car photos
- Knowledge Bases: Wikipedia for historical/educational content
- Video Search: YouTube for reviews and demonstrations
"""

import time
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from ddgs import DDGS
import wikipedia
from youtubesearchpython import VideosSearch


@tool
def car_web_search(query: str) -> str:
    """
    Search the web for current car information using DuckDuckGo.
    
    Use this tool for:
    - Current car specifications (horsepower, torque, 0-60 times)
    - Car prices and deals
    - Recent car reviews and news
    - Car comparisons
    - Any factual, up-to-date car data
    
    Args:
        query: Search query like 'Toyota Supra 2024 specs horsepower'
    
    Returns:
        Search results with car information
    """
    try:
        search = DuckDuckGoSearchRun()
        enhanced_query = f"{query} car specifications reviews"
        result = search.run(enhanced_query)
        # Ensure we return a string
        if isinstance(result, list):
            return "\n".join(str(r) for r in result)
        return str(result)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def car_image_search(query: str) -> str:
    """
    Search for car images and return image URLs that can be displayed.
    
    ALWAYS use this tool when user asks for:
    - "show me images of..."
    - "pictures of..."
    - "what does X look like"
    - "photos of..."
    - Any request to SEE a car visually
    
    Args:
        query: Car to search images for like 'Honda Civic Type R' or 'Ferrari 488'
    
    Returns:
        Markdown formatted images that will display in the chat
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ddgs = DDGS()
            results = list(ddgs.images(f"{query} car", max_results=6))
            
            if not results:
                return f"No images found for '{query}'"
            
            output = f"**📸 Images of {query}:**\n\n"
            
            for i, img in enumerate(results[:6], 1):
                title = img.get('title', 'Car Image')[:50]
                image_url = img.get('image', '')
                
                if image_url:
                    # Return markdown image format
                    output += f"![{title}]({image_url})\n\n"
            
            return output
            
        except Exception as e:
            error_msg = str(e)
            if "Ratelimit" in error_msg and attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
            return f"Image search error: {error_msg}. Try again in a moment."


@tool
def car_wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for car history, brand information, and detailed background.
    
    Use this tool for:
    - Car brand/manufacturer history (e.g., "history of Ford")
    - Classic car information
    - Car model heritage and generations
    - Technical explanations (e.g., "how does turbocharging work")
    - Car racing history
    
    Args:
        query: Topic to search like 'Ford Mustang history' or 'V8 engine'
    
    Returns:
        Wikipedia summary about the topic
    """
    try:
        # Search for relevant articles
        search_results = wikipedia.search(query + " car automobile", results=3)
        
        if not search_results:
            return f"No Wikipedia articles found for '{query}'"
        
        # Try to get summary from the most relevant result
        for result in search_results:
            try:
                summary = wikipedia.summary(result, sentences=5)
                return f"**{result}**\n\n{summary}"
            except wikipedia.DisambiguationError as e:
                # If disambiguation, try the first option
                if e.options:
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=5)
                        return f"**{e.options[0]}**\n\n{summary}"
                    except:
                        continue
            except wikipedia.PageError:
                continue
        
        return f"Could not retrieve Wikipedia content for '{query}'"
        
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


@tool
def youtube_car_videos(query: str) -> str:
    """
    Search YouTube for car videos, reviews, and demonstrations.
    
    Use this tool when user asks for:
    - Car video reviews
    - Car walkarounds or tours
    - Acceleration tests or performance videos
    - How-to guides for cars
    - Car comparisons in video format
    
    Args:
        query: Search query like 'BMW M3 2024 review' or 'Tesla Model S Plaid acceleration'
    
    Returns:
        List of relevant YouTube videos with titles and links
    """
    try:
        # Search YouTube
        search = VideosSearch(f"{query} car review", limit=5)
        results = search.result()
        
        if not results or 'result' not in results or not results['result']:
            return f"No YouTube videos found for '{query}'"
        
        videos = results['result']
        output = f"**🎬 YouTube Videos for '{query}':**\n\n"
        
        for i, video in enumerate(videos, 1):
            title = video.get('title', 'Unknown Title')
            channel = video.get('channel', {}).get('name', 'Unknown Channel')
            duration = video.get('duration', 'N/A')
            views = video.get('viewCount', {}).get('short', 'N/A views')
            link = video.get('link', '')
            
            output += f"{i}. **{title}**\n"
            output += f"   📺 Channel: {channel}\n"
            output += f"   ⏱️ Duration: {duration} | 👁️ {views}\n"
            output += f"   🔗 {link}\n\n"
        
        return output
        
    except Exception as e:
        return f"YouTube search error: {str(e)}"
