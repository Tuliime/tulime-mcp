Here's a **comprehensive `README.md`** for your project **Tulime-mcp**, including setup instructions with a virtual environment, `.env` configuration, and running the script.

---

````markdown
# 🌾 Tulime-mcp

Tulime-mcp is an AI-driven scraping assistant powered by MCP server and LangGraph, designed to extract complex data from the web using multiple tools in sequence.

## 🚀 Features

- Uses MCP (Multi-Client Proxy) to route scraping requests
- LangGraph-based agent to coordinate tool usage
- Anthropic Claude 3.5 integration for intelligent task execution
- Secure `.env`-based configuration

---

## 🧰 Requirements

- Python 3.9+
- `npx` (Node.js, for MCP server execution)
- Access to:
  - [Bright Data](https://brightdata.com/) or compatible proxy provider
  - [Anthropic API](https://www.anthropic.com/)

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Tuliime/tulime-mcp
cd tulime-mcp
```
````

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory with the following content:

```env
API_TOKEN=your_api_token_here
BROWSER_AUTH=your_browser_auth_token_here
WEB_UNLOCKER_ZONE=your_web_unlocker_zone_id_here
```

> ⚠️ **Important:** Do not share or commit this file publicly.

---

## 📂 Project Structure

```
tulime-mcp/
│
├── main.py                # Main entrypoint with chat + scraping logic
├── .env                   # Environment config (excluded from version control)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Ensures .env, .venv, etc. are ignored
```

---

## 🧠 How It Works

- Starts a `stdio_client` using MCP parameters
- Establishes a `ClientSession` for interaction
- Loads LangChain-compatible scraping tools via MCP adapters
- Uses Claude 3.5 to run tools in sequence

---

## 🏃 Running the Project

Once everything is set up:

```bash
python main.py
```
