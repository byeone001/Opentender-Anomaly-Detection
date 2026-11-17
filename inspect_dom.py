"""
Inspect DOM of radio buttons in Streamlit
"""
import urllib.request
import json
import asyncio
import websockets

async def get_dom():
    with urllib.request.urlopen('http://127.0.0.1:9222/json') as r:
        tabs = json.loads(r.read().decode())
    ws_url = tabs[0]['webSocketDebuggerUrl']
    async with websockets.connect(ws_url) as ws:
        js = """
        (function() {
            var labels = document.querySelectorAll('section[data-testid="stSidebar"] div[role="radiogroup"] label');
            var res = [];
            labels.forEach(function(l) {
                var items = [];
                for (var i = 0; i < l.children.length; i++) {
                    var c = l.children[i];
                    items.push({
                        tagName: c.tagName,
                        className: c.className,
                        testId: c.getAttribute('data-testid'),
                        role: c.getAttribute('role'),
                        ariaLabel: c.getAttribute('aria-label')
                    });
                }
                res.push({
                    text: l.innerText,
                    children: items,
                    html: l.outerHTML.substring(0, 400)
                });
            });
            return JSON.stringify(res);
        })()
        """
        await ws.send(json.dumps({'id': 1, 'method': 'Runtime.evaluate', 'params': {'expression': js}}))
        raw = await ws.recv()
        print('DOM result:', json.loads(raw)['result']['result']['value'])

if __name__ == '__main__':
    try:
        asyncio.run(get_dom())
    except Exception as e:
        print('Error:', e)
