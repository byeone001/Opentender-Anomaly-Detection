"""
Inspect exact radio element and Vega/Altair element in Chrome DOM
"""
import urllib.request
import json
import asyncio
import websockets

async def inspect():
    with urllib.request.urlopen("http://127.0.0.1:9222/json") as r:
        tabs = json.loads(r.read().decode())
    ws_url = tabs[0]["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url) as ws:
        js = """
        (function() {
            var firstLabel = document.querySelector('section[data-testid="stSidebar"] div[role="radiogroup"] label');
            var radioDiv = firstLabel ? firstLabel.firstElementChild.outerHTML : 'none';
            var vegaCharts = document.querySelectorAll('.vega-embed, [data-testid="stVegaLiteChart"]').length;
            var errors = [];
            window.addEventListener('error', function(e) { errors.push(e.message); });
            return JSON.stringify({
                labelHtml: firstLabel ? firstLabel.outerHTML : 'none',
                firstElementHtml: radioDiv,
                vegaChartsCount: vegaCharts
            });
        })()
        """
        await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": js}}))
        raw = await ws.recv()
        print("INSPECT RESULT:", json.loads(raw)["result"]["result"]["value"])

if __name__ == '__main__':
    asyncio.run(inspect())
