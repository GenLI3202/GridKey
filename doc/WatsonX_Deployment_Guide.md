# WatsonX Orchestrate Deployment Guide (Module D)

> **Purpose**: Connect your local GridKey Optimizer Service to IBM WatsonX Orchestrate as an AI Agent Skill.
> **Based on**: [Module E Integration Plan](../../wtsx_hack_gridkey/dev_doc/dev_plan/implemented/E_agent_layer_summary.md)

## 1. Prerequisites

-   **GridKey Service Running**: Your local FastAPI server (`startup.bat` or `startup.sh` is running).
-   **ngrok Installed**: Download from [ngrok.com](https://ngrok.com) (Free account is enough).
-   **WatsonX Account**: Access to [IBM WatsonX Orchestrate](https://watsonx.ai).

## 2. Step-by-Step Deployment

### Step 1: Create a Public Tunnel (The Bridge)
WatsonX is on the cloud, your service is on `localhost`. We need a bridge.

### Step 1: Create a Public Tunnel (The Bridge)

WatsonX is on the cloud, but your service is running inside your laptop (`localhost`). We need to build a "bridge" so WatsonX can see you.

**1. Start your API Server (Window A)**
1.  Open your file explorer to the project folder `GridKey`.
2.  Find the `startup.bat` file we created earlier.
3.  **Double-click** it.
4.  A black window (Terminal) will pop up showing logs like `Uvicorn running on...`.
5.  **DO NOT CLOSE THIS WINDOW.** Minimize it if you want, but it must stay open for your server to work.

**2. Start ngrok (Window B)**
1.  Open a **NEW** terminal window (Press `Win + R`, type `cmd`, hit Enter).
2.  In this new black window, type `ngrok http 8000` and hit Enter.
    *   *Note: If it says 'command not found', you need to download ngrok from ngrok.com and unzip it first.*
3.  You will see a screen with a "Forwarding" line, something like:
    `Forwarding   https://a1b2-c3d4.ngrok-free.app  ->  http://localhost:8000`
4.  **Copy that HTTPS URL** (`https://...ngrok-free.app`). This is your bridge address.
5.  **DO NOT CLOSE THIS WINDOW EITHER.** Both Window A and Window B must keep running.

### Step 2: Prepare the API Contract (The Map)
WatsonX needs to know "where" to call your API.

1.  **Download OpenAPI Spec**:
    Go to `http://localhost:8000/openapi.json` and save it as `gridkey_openapi.json`.

2.  **Modify the Server URL**:
    Open `gridkey_openapi.json` in your editor. Find the `servers` list (or add it if missing) at the top level:
    ```json
    {
      "openapi": "3.1.0",
      "info": { "title": "GridKey Optimizer", "version": "1.0.0" },
      "servers": [
        {
          "url": "https://YOUR-NGROK-ID.ngrok-free.app",
          "description": "Ngrok Public Tunnel"
        }
      ],
      "paths": ...
    }
    ```
    > **CRITICAL**: Replace the URL with *your actual* ngrok URL from Step 1.

### Step 3: Register Skill in WatsonX (The Brain)

1.  Log in to **IBM WatsonX Orchestrate**.
2.  Navigate to **Skills** (or Chat -> Add Skills).
3.  Click **Add skills** -> **From files** (Import OpenAPI).
4.  Upload your modified `gridkey_openapi.json`.
5.  **Skill Configuration**:
    *   It should detect endpoints like `/api/v1/optimize`.
    *   You might need to "Select" the operations you want to expose (e.g., `optimize`).
6.  **Activate**: Click "Save" / "Activate".

### Step 4: Test with Natural Language

Now go to the **Agent Chat** interface in WatsonX.

**Try asking:**
> "Run battery optimization for me."

(The agent might ask you for JSON details if the input is complex. In a real integration, the Agent would fetch Weather/Price data first and pass it to you automatically).

---

## 3. Troubleshooting & "Gotchas"

| Issue | Solution |
| :--- | :--- |
| **ngrok URL Expired** | Free ngrok URLs change every time you restart. You MUST update `gridkey_openapi.json` and **Re-import** the skill in WatsonX every time you restart ngrok. |
| **Validation Error** | WatsonX is strict about OpenAPI specs. If it complains, try deleting the `/health` endpoint from the JSON file, keeping only `/api/v1/optimize`. |
| **Timeout** | Optimization (MILP) might take >10s. Ensure WatsonX skill timeout settings (if accessible) allow for 30s-60s. |

## 4. Advanced: The "Full Loop" (Module A+B+D)

According to `E_agent_layer_summary.md`, the ultimate goal is:
1. Agent calls **Weather Skill** (Module A).
2. Agent calls **Price Skill** (Module B).
3. Agent passes those outputs into **Optimizer Skill** (Module D).

To support this transparency, ensure your `OptimizationInput` model in `models.py` matches what the Agent receives from A and B. (See `Integration_Guide.md`).
