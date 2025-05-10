import asyncio
import json
import os
import re
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# News data structure
class News:
    def __init__(
        self,
        title: str,
        description: str,
        category: str,
        source: str,
        source_url: str,
        image_url: str,
        posted_at: str,
    ):
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.category = category
        self.source = source
        self.source_url = source_url
        self.image_url = image_url
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert News object to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "source": self.source,
            "sourceUrl": self.source_url,
            "imageUrl": self.image_url,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at
        }

# Configuration for target news sites - simplified to main domains
UGANDA_NEWS_SITES = [
    {
        "name": "New Vision",
        "url": "https://www.newvision.co.ug"
    },
    {
        "name": "Daily Monitor",
        "url": "https://www.monitor.co.ug"
    },
    {
        "name": "UBC",
        "url": "https://ubc.go.ug"
    },
    {
        "name": "NTV Uganda",
        "url": "https://www.ntv.co.ug"
    },
    {
        "name": "NBS",
        "url": "https://nbs.ug"
    },
    {
        "name": "Nile Post",
        "url": "https://nilepost.co.ug"
    }
]

# Output directory for scraped data
OUTPUT_DIR = Path("./uganda_agriculture_news")

# Regular expressions for article detection
ARTICLE_PATTERNS = [
    r"/category/[a-z-]+/[a-z-]+/[a-z0-9-]+",
    r"/uganda/[a-z-]+/[a-z-]+/[a-z0-9-]+-[0-9]+",
    r"/[0-9]{4}/[0-9]{2}/[0-9]{2}/[a-z0-9-]+",
    r"/category/[a-z-]+/[a-z0-9-]+",
    r"/[0-9]{4}/[0-9]{2}/[a-z0-9-]+"
]

# Combined pattern for article detection
ARTICLE_PATTERN = re.compile("|".join(ARTICLE_PATTERNS))

class UgandaAgricultureScraper:
    def __init__(self):
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set up model
        self.model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
        
        # Server parameters
        self.server_params = StdioServerParameters(
            command="npx",
            env={
                "API_TOKEN": os.getenv("BRIGHTDATA_API_TOKEN"),
                "BROWSER_AUTH": os.getenv("BRIGHTDATA_BROWSER_AUTH"),
                "WEB_UNLOCKER_ZONE": os.getenv("BRIGHTDATA_WEB_UNLOCKER_ZONE"),
            },
            args=["@brightdata/mcp"]
        )
        
        # Track processed articles to avoid duplicates
        self.processed_urls = set()
        
        # Statistics
        self.stats = {
            "articles_found": 0,
            "agriculture_articles": 0,
            "sites_processed": 0
        }

    async def scrape_all_sites(self):
        """Main function to scrape all configured sites"""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(self.model, tools)
                
                # Process each news site
                for site in UGANDA_NEWS_SITES:
                    self.stats["sites_processed"] += 1
                    logger.info(f"Processing site: {site['name']}")
                    
                    # Scrape main URL
                    await self.process_site(agent, site["name"], site["url"])
                
                logger.info("Scraping complete!")
                logger.info(f"Statistics: {json.dumps(self.stats, indent=2)}")

    async def process_site(self, agent, site_name: str, url: str):
        """Process a single site URL to extract articles"""
        # First, navigate the site to find agriculture-related sections and articles
        links_prompt = f"""
        Visit the URL {url} and perform the following tasks:
        
        1. First, explore the site navigation to identify any agriculture-related sections or categories
           (look for sections like "Agriculture", "Farming", "Rural Development", etc.)
        
        2. Search for and identify recent news articles related to agriculture in Uganda.
           Look for content about:
           - Farming practices
           - Crop production
           - Livestock
           - Agricultural policy
           - Irrigation
           - Food security
           - Agricultural innovation
           - Weather impacts on farming
           - Agricultural markets and prices
           - Any other agriculture-related topics
        
        3. Extract links to these agriculture articles from both the navigation sections and any article listings.
        
        Return a JSON array of objects with the following structure:
        {{
            "url": "full article URL",
            "title": "article title or link text",
            "description": "brief description if available"
        }}
        
        Only include links that are specifically about agriculture or farming topics in Uganda.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert web scraper focusing on agriculture news from Uganda. You can navigate websites, search for relevant content, and extract structured data."},
            {"role": "user", "content": links_prompt}
        ]
        
        try:
            links_response = await agent.ainvoke({"messages": messages})
            links_content = links_response.get("content", "")
            
            # Log the raw response
            logger.info(f"Raw agent response for {url}:\n{links_content}")
            
            # Extract JSON from the response
            match = re.search(r'```json\n(.*?)```', links_content, re.DOTALL)
            if match:
                links_json = match.group(1)
            else:
                # Try finding JSON without code blocks
                match = re.search(r'\[\s*{\s*"url"', links_content)
                if match:
                    links_json = links_content[match.start():]
                else:
                    logger.error(f"Could not extract JSON from response for {url}")
                    return
            
            try:
                article_links = json.loads(links_json)
                if not isinstance(article_links, list):
                    logger.error(f"Expected list but got {type(article_links)} for {url}")
                    return
                
                logger.info(f"Found {len(article_links)} agriculture-related article links on {url}")
                self.stats["articles_found"] += len(article_links)
                
                # Process each article link
                for article in article_links:
                    article_url = article.get("url")
                    if not article_url or article_url in self.processed_urls:
                        continue
                    
                    # Add to processed set
                    self.processed_urls.add(article_url)
                    
                    # Check if it looks like an article
                    if not ARTICLE_PATTERN.search(article_url):
                        continue
                    
                    # Process the article
                    await self.process_article(agent, site_name, article_url)
                    
                    # Small delay to prevent overwhelming the service
                    await asyncio.sleep(1)
                    
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from response for {url}")
                logger.error(f"Response content: {links_content[:200]}...")
        
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")

    async def process_article(self, agent, site_name: str, article_url: str):
        """Process a single article to extract content"""
        article_prompt = f"""
        Visit the URL {article_url} and extract information from this agriculture-related news article.
        
        Extract the following:
        
        1. Article title
        2. Full article text content
        3. Publication date
        4. Main image URL (if available)
        5. Why this article is relevant to agriculture in Uganda (briefly explain what agricultural topics it covers)
        
        Return the data in JSON format with these fields:
        {{
            "title": "article title",
            "content": "full article text",
            "date": "publication date",
            "image_url": "main image URL or empty string if none",
            "agriculture_relevance": "explanation of how this relates to agriculture"
        }}
        
        Make sure to capture the complete article text, not just the first paragraph.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert web scraper focusing on agriculture news from Uganda."},
            {"role": "user", "content": article_prompt}
        ]
        
        try:
            article_response = await agent.ainvoke({"messages": messages})
            article_content = article_response.get("content", "")
            
            # Log the raw response
            logger.info(f"Raw agent response for article {article_url}:\n{article_content}")
            
            # Extract JSON from the response
            match = re.search(r'```json\n(.*?)```', article_content, re.DOTALL)
            if match:
                article_json = match.group(1)
            else:
                # Try finding JSON without code blocks
                match = re.search(r'{\s*"title"', article_content)
                if match:
                    article_json = article_content[match.start():]
                else:
                    logger.error(f"Could not extract JSON from response for {article_url}")
                    return
            
            try:
                article_data = json.loads(article_json)
                
                # The agent has already verified this is agriculture-related
                self.stats["agriculture_articles"] += 1
                logger.info(f"Processing agriculture article: {article_data.get('title', 'Unknown')}")
                
                # Store the image URL directly
                image_url = article_data.get("image_url", "")
                
                # Create News object
                news = News(
                    title=article_data.get("title", ""),
                    description=article_data.get("content", ""),
                    category="Agriculture",
                    source=site_name,
                    source_url=article_url,
                    image_url=image_url,
                    posted_at=article_data.get("date", datetime.now().isoformat())
                )
                
                # Save to JSON file
                self.save_news_to_json(news)
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from response for {article_url}")
                logger.error(f"Response content: {article_content[:200]}...")
        
        except Exception as e:
            logger.error(f"Error processing article {article_url}: {str(e)}")

    def save_news_to_json(self, news: News):
        """Save news article to JSON file"""
        output_file = OUTPUT_DIR / f"{news.id}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(news.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved article to {output_file}")

async def main():
    scraper = UgandaAgricultureScraper()
    await scraper.scrape_all_sites()

if __name__ == "__main__":
    asyncio.run(main())