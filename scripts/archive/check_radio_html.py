import subprocess
import time
import json
import urllib.request
import asyncio
import websockets
import os

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
STREAMLIT_EXE = r".venv\Scripts\streamlit.exe"

proc_st = subprocess.Popen([
    STREAMLIT_EXE, "run", "app.py",
    "--server.headless", "true",
    "--server.port", "8501",
    "--server.address", "127.0.0.1"
])

for i in range(20):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=1) as r:
            if r.status == 200:
                break
    except Exception:
        time.sleep(1)

proc_chrome = subprocess.Popen([
    CHROME_PATH,
    "--headless=new",
    "--remote-debugging-port=9222",
    "http://127.0.0.1:8501"
])

time.sleep(3)

async def check():
    with urllib.request.urlopen("http://127.0.0.1:9222/json") as r:
        tabs = json.loads(r.read().decode())
    ws_url = tabs[0]["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url) as ws:
        await asyncio.sleep(12)
        js = """
        (function() {
            var radios = document.querySelectorAll('div[data-testid="stRadio"] *');
            var tags = [];
            radios.forEach(function(el) {
                tags.push(el.tagName + (el.className ? '.' + el.className.split(' ').join('.') : ''));
            });
            var radioContainer = document.querySelector('div[data-testid="stRadio"]');
            return {
                containerHTML: radioContainer ? radioContainer.innerHTML.substring(0, 1000) : "NO CONTAINER"
            };
        })()
        """
        await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": js}}))
        raw = await ws.recv()
        print("RADIO CONTAINER:", json.loads(raw)["result"]["result"]["value"])

try:
    asyncio.run(check())
finally:
    proc_chrome.terminate()
    proc_st.terminate()
