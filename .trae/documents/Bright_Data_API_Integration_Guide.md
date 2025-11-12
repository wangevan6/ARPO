# Bright Data API Integration Guide for ARPO

**Date**: 2025-11-10
**Purpose**: Understand how ARPO makes Bright Data API calls for Bing Search
**Location**: `verl_arpo_entropy/verl/workers/rollout/tools/search_tool.py`

---

## API Type Comparison

### ❌ What You DON'T Need (Scraping Browser)
The code you showed uses **Scraping Browser** (Playwright/WebSocket):
```python
endpoint_url = f'wss://{AUTH}@brd.superproxy.io:9222'  # WebSocket connection
browser = playwright.chromium.connect_over_cdp(endpoint_url)
```
**This is NOT what ARPO uses!**

### ✅ What ARPO Actually Uses (SERP API)
ARPO uses **Bright Data's REST API for Search Engine Results Page (SERP)**:
- **Endpoint**: `https://api.brightdata.com/request`
- **Method**: HTTP POST
- **Authentication**: Bearer token
- **Purpose**: Get Bing search results as JSON

---

## Exact API Call Trace

### Step-by-Step Flow

```mermaid
sequenceDiagram
    participant Model
    participant SearchTool
    participant Cache
    participant BrightData as Bright Data API
    participant Bing as Bing.com

    Model->>SearchTool: execute("What is Python?")
    SearchTool->>Cache: Check if "What is Python?" cached
    alt Cache Hit
        Cache-->>SearchTool: Return cached results
        SearchTool-->>Model: Return results
    else Cache Miss
        SearchTool->>BrightData: POST /request
        Note over SearchTool,BrightData: Headers: Authorization: Bearer {api_key}<br/>Payload: {zone, url, format}
        BrightData->>Bing: Fetch https://www.bing.com/search?q=What+is+Python
        Bing-->>BrightData: HTML/JSON response
        BrightData-->>SearchTool: JSON with organic results
        SearchTool->>SearchTool: Extract snippets
        SearchTool->>Cache: Store results (async)
        SearchTool-->>Model: Return formatted results
    end
```

### Code Trace (Line Numbers from search_tool.py)

#### 1. **Initialization** (Lines 30-78)

```python
def __init__(
    self,
    api_key: str,                    # ← YOUR API KEY HERE
    zone: str = "serp_api1",         # ← YOUR ZONE NAME HERE
    max_results: int = 10,
    result_length: int = 1000,
    location: str = "cn",            # Country code
    cache_file: Optional[str] = None,
    async_cache_write: bool = True,
    cache_refresh_interval: float = 15.0
):
    self._api_key = api_key          # Store API key
    self._zone = zone                # Store zone name
    ...
```

**What you need**:
- `api_key`: Your Bright Data API key (e.g., `"brd-customer-abc123-zone-xyz456"`)
- `zone`: Your zone name (e.g., `"serp_api1"` or `"residential"`)

#### 2. **Execute Search** (Lines 380-430)

```python
def execute(self, query: str, timeout: int = 60) -> str:
    # Clean query
    query = query.replace('"', '')

    # Check cache first
    with self._cache_lock:
        if query in self._cache:
            return self._cache[query]  # Return cached result

    # Make API request
    response = self._make_request(query, timeout)  # ← API CALL HERE

    # Parse and format results
    data = json.loads(response.text)
    result = self._extract_and_format_results(data)

    # Cache the result
    self._cache[query] = result
    self._save_cache()

    return result
```

#### 3. **Make API Request** (Lines 335-378) - **THE CORE API CALL**

```python
def _make_request(self, query: str, timeout: int) -> requests.Response:
    # Detect query language (Chinese vs English)
    lang_code, lang_confidence = langid.classify(query)
    if lang_code == 'zh':
        mkt, setLang = "zh-CN", "zh"
    else:
        mkt, setLang = "en-US", "en"

    # Build Bing search URL
    encoded_query = urlencode({
        "q": query,              # Search query
        "mkt": mkt,              # Market (zh-CN or en-US)
        "setLang": setLang       # Language
    })
    target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc={self._location}"

    # Prepare request headers
    headers = {
        "Authorization": f"Bearer {self._api_key}",  # ← YOUR API KEY USED HERE
        "Content-Type": "application/json"
    }

    # Prepare request payload
    payload = {
        "zone": self._zone,      # ← YOUR ZONE USED HERE
        "url": target_url,       # Bing URL to scrape
        "format": "raw"          # Return raw JSON
    }

    # Send POST request to Bright Data API
    return requests.post(
        "https://api.brightdata.com/request",  # ← BRIGHT DATA ENDPOINT
        headers=headers,
        json=payload,
        timeout=timeout
    )
```

**Breakdown**:
1. **Detect Language**: Auto-detect if query is Chinese or English
2. **Build Target URL**: Create Bing search URL with query
3. **Prepare Headers**: Add Authorization with Bearer token
4. **Prepare Payload**: Specify zone, target URL, and format
5. **Send POST Request**: To `https://api.brightdata.com/request`

#### 4. **Example Request**

For query: `"What is Python?"`

**HTTP Request**:
```http
POST https://api.brightdata.com/request HTTP/1.1
Content-Type: application/json
Authorization: Bearer brd-customer-abc123-zone-xyz456

{
  "zone": "serp_api1",
  "url": "https://www.bing.com/search?q=What+is+Python&mkt=en-US&setLang=en&brd_json=1&cc=cn",
  "format": "raw"
}
```

**Expected Response**:
```json
{
  "organic": [
    {
      "title": "Python (programming language) - Wikipedia",
      "description": "Python is a high-level, general-purpose programming language...",
      "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"
    },
    {
      "title": "Welcome to Python.org",
      "description": "The official home of the Python Programming Language...",
      "url": "https://www.python.org/"
    }
  ]
}
```

#### 5. **Format Results** (Lines 432-477)

```python
def _extract_and_format_results(self, data: Dict) -> str:
    # Extract snippets from organic results
    chunk_content_list = []
    for result in data['organic']:
        snippet = result.get('description', '').strip()
        if snippet and snippet not in seen_snippets:
            chunk_content_list.append(snippet)

    # Format as "Page 1: ...", "Page 2: ...", etc.
    formatted = []
    for idx, snippet in enumerate(chunk_content_list[:self._max_results], 1):
        formatted.append(f"Page {idx}: {snippet[:self._result_length]}")

    return "\n".join(formatted)
```

**Final Output** (what the model sees):
```
Page 1: Python is a high-level, general-purpose programming language...
Page 2: The official home of the Python Programming Language...
Page 3: Python is an interpreted, object-oriented, high-level programming language...
```

---

## How to Get Your API Credentials

### Step 1: Sign Up for Bright Data

1. Go to: https://brightdata.com/
2. Create an account (free trial available)
3. Verify your email

### Step 2: Create a Zone

1. Log in to Bright Data dashboard
2. Navigate to **Zones** → **Create Zone**
3. Select **SERP API** (NOT Scraping Browser!)
4. Choose:
   - **Zone Type**: SERP API
   - **Search Engine**: Bing
   - **Country**: Your target country (US, CN, etc.)
5. Click **Create Zone**

### Step 3: Get Credentials

After creating the zone, you'll see:

```
Zone Name: serp_api1 (or your custom name)
API Token: brd-customer-hl_YOUR_CUSTOMER_ID-zone-YOUR_ZONE_NAME:YOUR_PASSWORD
```

**Example**:
```
Zone: serp_api1
API Token: brd-customer-hl_b374325b-zone-serp_api1:xp0ziu3gywj8
```

**Split into parts**:
- `api_key`: `brd-customer-hl_b374325b-zone-serp_api1:xp0ziu3gywj8`
- `zone`: `serp_api1`

---

## Where to Add Your Credentials in ARPO

You need to update **3 locations**:

### Location 1: Training Script

**File**: `ARPO/scripts/ARPO_3B_Toy_Local.sh` (or your training script)

```bash
# UPDATE THESE LINES:
BING_API_KEY="brd-customer-hl_b374325b-zone-serp_api1:xp0ziu3gywj8"  # ← YOUR FULL API TOKEN
BING_ZONE="serp_api1"  # ← YOUR ZONE NAME
```

### Location 2: Hydra Config

**File**: `ARPO/scripts/config/ppo_trainer.yaml`

Find the search tool section (around line 300):
```yaml
actor_rollout_ref:
  rollout:
    tools:
      tool_instances:
        search:
          class_path: verl.workers.rollout.tools.search_tool.BingSearchTool
          params:
            api_key: YOUR_API_KEY_HERE  # ← UPDATE THIS
            zone: YOUR_ZONE_HERE         # ← UPDATE THIS
            max_results: 10
            result_length: 1000
            location: us
            cache_file: search_cache/search_cache.json
```

**Update to**:
```yaml
            api_key: brd-customer-hl_b374325b-zone-serp_api1:xp0ziu3gywj8
            zone: serp_api1
```

### Location 3: Tool Config (Optional)

**File**: `ARPO/verl_arpo_entropy/verl/workers/rollout/tools/config.yaml`

If this file exists, update it similarly:
```yaml
tools:
  tool_instances:
    search:
      params:
        api_key: brd-customer-hl_b374325b-zone-serp_api1:xp0ziu3gywj8
        zone: serp_api1
```

---

## Testing Your API Credentials

### Quick Test Script

Create: `ARPO/test_bright_data.py`

```python
#!/usr/bin/env python3
import requests
import json

# YOUR CREDENTIALS
API_KEY = "brd-customer-hl_b374325b-zone-serp_api1:xp0ziu3gywj8"  # ← UPDATE
ZONE = "serp_api1"  # ← UPDATE

# Test query
query = "What is Python programming"
target_url = f"https://www.bing.com/search?q={query}&mkt=en-US&setLang=en&brd_json=1&cc=us"

# Prepare request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "zone": ZONE,
    "url": target_url,
    "format": "raw"
}

# Send request
print(f"Testing Bright Data API...")
print(f"Query: {query}")
print(f"Zone: {ZONE}")
print()

response = requests.post(
    "https://api.brightdata.com/request",
    headers=headers,
    json=payload,
    timeout=30
)

print(f"Status Code: {response.status_code}")
print()

if response.status_code == 200:
    data = json.loads(response.text)
    print("✅ API Test SUCCESSFUL!")
    print()
    print("Results:")
    if 'organic' in data:
        for i, result in enumerate(data['organic'][:3], 1):
            print(f"\n{i}. {result.get('title', 'No title')}")
            print(f"   {result.get('description', 'No description')[:100]}...")
    else:
        print("No organic results found")
        print(f"Response: {json.dumps(data, indent=2)}")
else:
    print("❌ API Test FAILED!")
    print(f"Error: {response.text}")
```

**Run test**:
```bash
cd C:/Users/user/Projects/ARPO/ARPO
python test_bright_data.py
```

**Expected Output**:
```
Testing Bright Data API...
Query: What is Python programming
Zone: serp_api1

Status Code: 200

✅ API Test SUCCESSFUL!

Results:

1. Python (programming language) - Wikipedia
   Python is a high-level, general-purpose programming language. Its design philosophy emphasizes ...

2. Welcome to Python.org
   The official home of the Python Programming Language...

3. What is Python? Executive Summary
   Python is an interpreted, object-oriented, high-level programming language with dynamic semantics...
```

---

## API Cost Estimation

**Bright Data SERP API Pricing** (approximate):
- **Free Trial**: 1,000-5,000 requests (check current offer)
- **Pay-as-you-go**: $3-5 per 1,000 requests
- **Monthly Plans**: $500-2000 for 100K-1M requests

**ARPO Training Usage**:
- **Toy Training** (100 samples, 1 epoch): ~50-200 search queries
- **Production Training** (10K samples, 2 epochs): ~2,000-8,000 queries
- **Estimated Cost**: $6-40 per production training run

**Optimization via Caching**:
- ARPO uses aggressive caching (`search_cache.json`)
- Retraining on same dataset: ~90% cache hit rate
- Multiple experiments: Share cache across runs

---

## Common API Errors

### Error 1: 401 Unauthorized
```
HTTP 401: Unauthorized
```
**Cause**: Wrong API key
**Fix**: Double-check your API token from Bright Data dashboard

### Error 2: 403 Forbidden
```
HTTP 403: Forbidden - Zone not found
```
**Cause**: Wrong zone name
**Fix**: Verify zone name matches your Bright Data zone

### Error 3: 429 Rate Limit
```
HTTP 429: Too Many Requests
```
**Cause**: Exceeded rate limit
**Fix**: Slow down requests or upgrade plan

### Error 4: Timeout
```
Bing search failed: HTTPSConnectionPool...timed out
```
**Cause**: Network issue or API down
**Fix**: Increase timeout, check internet connection

---

## API Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ARPO RL Training                         │
│                                                             │
│  Model generates: "What is the capital of France?"         │
│                           ↓                                 │
│  Rollout detects: <search>capital of France</search>       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              BingSearchTool.execute()                       │
│                                                             │
│  1. Check cache for "capital of France"                    │
│     - Cache hit? → Return cached result                    │
│     - Cache miss? → Continue to API call                   │
│                                                             │
│  2. _make_request()                                        │
│     - Detect language (English)                            │
│     - Build URL: bing.com/search?q=capital+of+France       │
│     - Prepare headers: Authorization: Bearer {api_key}     │
│     - Prepare payload: {zone, url, format}                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓ HTTP POST
┌─────────────────────────────────────────────────────────────┐
│          https://api.brightdata.com/request                 │
│                                                             │
│  Receives:                                                  │
│  {                                                          │
│    "zone": "serp_api1",                                    │
│    "url": "https://www.bing.com/search?q=...",           │
│    "format": "raw"                                         │
│  }                                                          │
│                                                             │
│  Forwards request to → Bing.com                            │
│  Receives HTML/JSON ← Bing.com                             │
│  Returns JSON response                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓ JSON Response
┌─────────────────────────────────────────────────────────────┐
│              BingSearchTool.execute()                       │
│                                                             │
│  3. _extract_and_format_results()                          │
│     - Parse JSON response                                  │
│     - Extract organic results                              │
│     - Format as "Page 1: ...", "Page 2: ..."              │
│                                                             │
│  4. Cache result                                           │
│     - Store in memory cache                                │
│     - Async write to search_cache.json                     │
│                                                             │
│  5. Return formatted results                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓ Formatted Results
┌─────────────────────────────────────────────────────────────┐
│                    ARPO RL Training                         │
│                                                             │
│  Rollout appends:                                          │
│  <result>                                                  │
│  Page 1: Paris is the capital of France...                │
│  Page 2: France capital city facts...                     │
│  </result>                                                 │
│                                                             │
│  Model continues: "The capital of France is Paris."        │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Differences from Scraping Browser

| Feature | Scraping Browser (Your Example) | SERP API (ARPO Uses) |
|---------|--------------------------------|----------------------|
| **Protocol** | WebSocket (wss://) | HTTP REST (POST) |
| **Connection** | Persistent browser | Request/response |
| **Endpoint** | `brd.superproxy.io:9222` | `api.brightdata.com/request` |
| **Auth** | In connection string | Bearer token in headers |
| **Use Case** | Full web scraping | Search results only |
| **Dependencies** | Playwright, browser | requests library only |
| **ARPO Needs?** | ❌ NO | ✅ YES |

---

## Summary

### What ARPO Does:
1. Model generates `<search>query</search>` tag
2. SearchTool checks cache for query
3. If not cached, sends POST to `https://api.brightdata.com/request`
4. Bright Data fetches Bing search results
5. SearchTool extracts top snippets
6. Returns formatted text: "Page 1: ..., Page 2: ..."
7. Model uses results to generate answer

### What You Need:
1. **Bright Data Account** with SERP API access
2. **API Token** (full string: `brd-customer-...-zone-...:password`)
3. **Zone Name** (e.g., `serp_api1`)
4. Update 3 locations: training script, ppo_trainer.yaml, tool config

### Next Steps:
1. ✅ Get Bright Data credentials
2. ✅ Run test script to verify API works
3. ✅ Update all 3 configuration locations
4. ✅ Ready to start RL training!

---

**Document Version**: 1.0
**Last Updated**: 2025-11-10
**Related**: `RL_Training_Quickstart.md`, `LLaMA_Factory_SFT_Training_Process_Guide.md`
