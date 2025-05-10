import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

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

# Configuration for target news sites
UGANDA_NEWS_SITES = [
    # {
    #     "name": "New Vision", //To be removed or implement logging in for tulime
    #     "url": "https://www.newvision.co.ug/category/business/agriculture",
    #     "alternative_urls": [
    #         "https://www.newvision.co.ug/search?q=agriculture",
    #         "https://www.newvision.co.ug/search?q=farming"
    #     ]
    # },
    # {
    #     "name": "Daily Monitor", //To be removed
    #     "url": "https://www.monitor.co.ug/uganda/business/farming",
    #     "alternative_urls": [
    #         "https://www.monitor.co.ug/uganda/magazines/farming",
    #         "https://www.monitor.co.ug/uganda/search?q=agriculture"
    #     ]
    # },
    # {
    #     "name": "UBC", //To be removed
    #     "url": "https://ubc.go.ug/category/agriculture",
    #     "alternative_urls": [
    #         "https://ubc.go.ug/?s=agriculture",
    #         "https://ubc.go.ug/?s=farming"
    #     ]
    # },
    # {
    #     "name": "NTV Uganda",
    #     "url": "https://www.ntv.co.ug/category/news/agriculture",
    #     "alternative_urls": [
    #         "https://www.ntv.co.ug/search?q=agriculture",
    #         "https://www.ntv.co.ug/search?q=farming"
    #     ]
    # },
    # {
    #     "name": "NBS", //To be removed
    #     "url": "https://nbs.ug/tag/agriculture/",
    #     "alternative_urls": [
    #         "https://nbs.ug/tag/farming/",
    #         "https://nbs.ug/?s=agriculture"
    #     ]
    # },
    {
        "name": "Nile Post",
        "url": "https://nilepost.co.ug/agriculture/",
        # "alternative_urls": [
        #     "https://nilepost.co.ug/?s=agriculture",
        #     "https://nilepost.co.ug/?s=farming"
        # ]
            "alternative_urls": []
    }
]

# Agriculture-related keywords for content filtering
AGRICULTURE_KEYWORDS = [
    "agriculture", "farming", "crops", "livestock", "irrigation",
    "harvest", "maize", "coffee", "dairy", "cattle", "goats",
    "vegetables", "fertilizer", "agricultural", "farm", "farmer",
    "farmers", "seeds", "plantation", "soil", "poultry", "chickens",
    "agribusiness", "agritech", "agronomy", "agroforestry", "banana",
    "cassava", "cotton", "rice", "fish", "fisheries", "food security",
    "pesticide", "drought", "rain", "weather", "climate", "grain",
    "crop disease", "agricultural extension", "organic", "conservation",
    "cooperative", "sustainable", "subsistence", "commercial farming"
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
                    print(f"Processing site: {site['name']}")
                    
                    # Scrape main URL
                    await self.process_site(agent, site["name"], site["url"])
                    
                    # Scrape alternative URLs
                    for alt_url in site["alternative_urls"]:
                        await self.process_site(agent, site["name"], alt_url)
                
                print("Scraping complete!")
                print(f"Statistics: {json.dumps(self.stats, indent=2)}")

    async def process_site(self, agent, site_name: str, url: str):
        """Process a single site URL to extract articles"""
        # First, get all article links from the page
        links_prompt = f"""
        Visit the URL {url} and extract all links that appear to be news articles about agriculture.
        Focus on links that match these patterns:
        - Links with 'agriculture', 'farming', 'crops', 'livestock' in the URL
        - Links to article pages (not navigation, footer, or header links)
        
        Return a JSON array of objects with the following structure:
        {{
            "url": "full article URL",
            "title": "article title or link text"
        }}
        
        Only include links that appear to be actual news articles.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert web scraper focusing on agriculture news from Uganda."},
            {"role": "user", "content": links_prompt}
        ]
        
        try:
            links_response = await agent.ainvoke({"messages": messages})
            links_content = links_response.get("content", "")
            
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
                    print(f"Could not extract JSON from response for {url}")
                    return
            
            try:
                article_links = json.loads(links_json)
                if not isinstance(article_links, list):
                    print(f"Expected list but got {type(article_links)} for {url}")
                    return
                
                print(f"Found {len(article_links)} potential article links on {url}")
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
                print(f"Failed to parse JSON from response for {url}")
                print(f"Response content: {links_content[:200]}...")
        
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")

    async def process_article(self, agent, site_name: str, article_url: str):
        """Process a single article to extract content"""
        article_prompt = f"""
        Visit the URL {article_url} and extract the following information:
        
        1. Article title
        2. Full article text content
        3. Publication date
        4. Main image URL (if available)
        
        Return the data in JSON format with these fields:
        {{
            "title": "article title",
            "content": "full article text",
            "date": "publication date",
            "image_url": "main image URL or empty string if none"
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
                    print(f"Could not extract JSON from response for {article_url}")
                    return
            
            try:
                article_data = json.loads(article_json)
                
                # Check if this is agriculture-related content
                is_agriculture = self.is_agriculture_related(
                    article_data.get("title", ""), 
                    article_data.get("content", "")
                )
                
                if not is_agriculture:
                    print(f"Skipping non-agriculture article: {article_url}")
                    return
                
                self.stats["agriculture_articles"] += 1
                print(f"Processing agriculture article: {article_data.get('title', 'Unknown')}")
                
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
                print(f"Failed to parse JSON from response for {article_url}")
                print(f"Response content: {article_content[:200]}...")
        
        except Exception as e:
            print(f"Error processing article {article_url}: {str(e)}")

    def is_agriculture_related(self, title: str, content: str) -> bool:
        """Check if content is related to agriculture based on keywords"""
        text = (title + " " + content).lower()
        
        for keyword in AGRICULTURE_KEYWORDS:
            if keyword.lower() in text:
                return True
        
        return False

    def save_news_to_json(self, news: News):
        """Save news article to JSON file"""
        output_file = OUTPUT_DIR / f"{news.id}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(news.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"Saved article to {output_file}")

async def main():
    scraper = UgandaAgricultureScraper()
    await scraper.scrape_all_sites()

if __name__ == "__main__":
    asyncio.run(main())



