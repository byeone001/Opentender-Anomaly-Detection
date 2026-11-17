"""
Capture high-res screenshot when KPI cards and Vega charts are fully rendered.
"""

import subprocess
import time
import json
import urllib.request
import asyncio
import websockets
import os
import sys
import base64

STREAMLIT_EXE = r".venv\Scripts\streamlit.exe"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
output_png = os.path.abspath("screenshot_dashboard.png")

proc_st = subprocess.Popen([
    STREAMLIT_EXE, "run", "app.py",
    "--server.headless", "true",
    "--server.port", "8501",
    "--server.address", "127.0.0.1"
])

print("Menunggu Streamlit...")
for i in range(25):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=1) as r:
            if r.status == 200:
                print(f"Streamlit siap ({i+1}s)!")
                break
    except Exception:
        time.sleep(1)

proc_chrome = subprocess.Popen([
    CHROME_PATH,
    "--headless=new",
    "--disable-gpu",
    "--remote-debugging-port=9222",
    "--window-size=1440,1100",
    "http://127.0.0.1:8501"
])

time.sleep(2)

async def capture():
    with urllib.request.urlopen("http://127.0.0.1:9222/json") as resp:
        tabs = json.loads(resp.read().decode())
    
    ws_url = None
    for t in tabs:
        if t.get("type") == "page" and "8501" in t.get("url", ""):
            ws_url = t.get("webSocketDebuggerUrl")
            break
    if not ws_url and tabs:
        ws_url = tabs[0].get("webSocketDebuggerUrl")

    async with websockets.connect(ws_url) as ws:
        print("Menunggu render DOM (KPI cards & Vega)...")
        
        for sec in range(45):
            await asyncio.sleep(1)
            js = """
            (function() {
                var kpi = document.querySelectorAll('.kpi-card').length;
                var svg = document.querySelectorAll('[data-testid="stVegaLiteChart"] svg, .vega-embed svg').length;
                return {kpi: kpi, svg: svg};
            })()
            """
            msg = {"id": 100 + sec, "method": "Runtime.evaluate", "params": {"expression": js}}
            await ws.send(json.dumps(msg))
            res = json.loads(await ws.recv())
            val = res.get("result", {}).get("result", {}).get("value", {})
            kpi_cnt = val.get("kpi", 0) if isinstance(val, dict) else 0
            svg_cnt = val.get("svg", 0) if isinstance(val, dict) else 0
            print(f"  Detik {sec+1}: KPI cards={kpi_cnt}, Vega SVGs={svg_cnt}")
            if kpi_cnt >= 4 and svg_cnt >= 2:
                print(f"RENDER LENGKAP: {kpi_cnt} KPI cards dan {svg_cnt} charts!")
                break

        await asyncio.sleep(2)

        msg = {
            "id": 1,
            "method": "Page.captureScreenshot",
            "params": {"format": "png", "captureBeyondViewport": False}
        }
        await ws.send(json.dumps(msg))
        
        while True:
            raw = await ws.recv()
            data = json.loads(raw)
            if data.get("id") == 1:
                img_bytes = base64.b64decode(data["result"]["data"])
                with open(output_png, "wb") as f:
                    f.write(img_bytes)
                print(f"BERHASIL! Screenshot tersimpan: {output_png} ({len(img_bytes):,} bytes)")
                break

try:
    asyncio.run(capture())
finally:
    proc_chrome.terminate()
    proc_st.terminate()
    print("Selesai.")
