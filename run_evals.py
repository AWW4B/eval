"""
run_evals.py — NLP Assignment 5 Evaluation Suite
=================================================
Runs correctness, latency, and throughput evals against the
Daraz AI backend at localhost:8000.

Features:
  - LLM-as-judge: spawns judge_worker.py inside the Docker container
    (uses the same Qwen GGUF model but with a neutral evaluator system prompt)
  - SSE server on port 8765 so index.html can stream live progress
  - Saves eval_results.json and eval_report.md on completion

Usage:
    python run_evals.py
    # Open index.html served via: python -m http.server 8080
"""

import asyncio
import json
import subprocess
import time
import uuid
import statistics
import platform
import psutil
import websockets
import aiohttp
from data import RAG_QA_PAIRS, MULTI_TURN_DIALOGUES, TOOL_TEST_CASES, LATENCY_SCENARIOS

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL        = "http://localhost:8000"
WS_URL          = "ws://localhost:8000/ws/chat"
ADMIN_ID        = "admin_eval_user"
LATENCY_TRIALS  = 5
SSE_PORT        = 8765
DOCKER_CONTAINER = "hybrid-rag-agent-backend-1"

# ── SSE broadcast queue ───────────────────────────────────────────────────────
_sse_clients: list[asyncio.Queue] = []

async def sse_broadcast(event: dict):
    for q in list(_sse_clients):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

# ── SSE HTTP server (aiohttp) ─────────────────────────────────────────────────
async def _sse_handler(request):
    from aiohttp import web
    response = web.StreamResponse(headers={
        "Content-Type":                "text/event-stream",
        "Cache-Control":               "no-cache",
        "Access-Control-Allow-Origin": "*",
        "X-Accel-Buffering":           "no",
    })
    await response.prepare(request)
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_clients.append(q)
    try:
        while True:
            event = await q.get()
            data  = json.dumps(event, ensure_ascii=False)
            await response.write(f"data: {data}\n\n".encode())
            if event.get("type") == "complete":
                break
    except (ConnectionResetError, Exception):
        pass
    finally:
        if q in _sse_clients:
            _sse_clients.remove(q)
    return response


async def _index_handler(request):
    """Serve the eval dashboard HTML at /"""
    import os
    from aiohttp import web
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return web.FileResponse(html_path)


async def _results_handler(request):
    """Serve eval_results.json if it exists"""
    import os
    from aiohttp import web
    json_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    if os.path.exists(json_path):
        return web.FileResponse(json_path, headers={"Access-Control-Allow-Origin": "*"})
    return web.Response(status=404, text="No results yet")


async def start_sse_server():
    from aiohttp import web
    app = web.Application()
    app.router.add_get("/", _index_handler)
    app.router.add_get("/events", _sse_handler)
    app.router.add_get("/eval_results.json", _results_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", SSE_PORT)
    await site.start()
    print(f"[SSE] Dashboard → http://localhost:{SSE_PORT}/")
    print(f"[SSE] Events    → http://localhost:{SSE_PORT}/events")
    return runner

# ── System info ───────────────────────────────────────────────────────────────
async def get_system_info() -> dict:
    return {
        "os":             platform.system(),
        "processor":      platform.processor(),
        "ram":            f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "python_version": platform.python_version(),
    }

# ── Docker-based LLM Judge ────────────────────────────────────────────────────
class DockerJudge:
    """
    Spawns judge_worker.py inside the already-running backend Docker container.
    The worker loads the Qwen GGUF model with a neutral evaluator system prompt
    and accepts JSON-line queries over stdin, returning JSON-line verdicts.
    """
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = asyncio.Lock()

    async def start(self):
        import os
        worker_path = os.path.join(os.path.dirname(__file__), "judge_worker.py")

        print("[judge] Copying judge_worker.py into Docker container...")
        loop = asyncio.get_running_loop()

        # Copy the script into the container at /tmp/judge_worker.py
        copy_proc = await loop.run_in_executor(None, lambda: subprocess.run(
            ["docker", "cp", worker_path, f"{DOCKER_CONTAINER}:/tmp/judge_worker.py"],
            capture_output=True,
        ))
        if copy_proc.returncode != 0:
            raise RuntimeError(f"Failed to copy judge_worker: {copy_proc.stderr.decode()}")

        print("[judge] Starting judge LLM inside Docker container...")
        # Now run it as a real file — stdin is free for JSON queries
        self._proc = await loop.run_in_executor(None, lambda: subprocess.Popen(
            ["docker", "exec", "-i", DOCKER_CONTAINER, "python3", "/tmp/judge_worker.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ))

        # Wait for "ready" signal (model loaded)
        print("[judge] Waiting for judge model to load (this may take ~60s)...")
        ready_line = await loop.run_in_executor(None, self._proc.stdout.readline)
        try:
            ready = json.loads(ready_line.decode().strip())
            if ready.get("ready"):
                print("[judge] Judge LLM ready ✓")
            else:
                print(f"[judge] Unexpected ready response: {ready_line}")
        except Exception as e:
            print(f"[judge] Parse error on ready signal: {e} | raw: {ready_line}")

    async def judge(self, question: str, expected: str, actual: str) -> bool:
        if self._proc is None:
            return False
        loop = asyncio.get_running_loop()
        async with self._lock:
            payload = json.dumps({"question": question, "expected": expected, "actual": actual})
            def _call():
                self._proc.stdin.write((payload + "\n").encode())
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                return json.loads(line.decode())
            result = await loop.run_in_executor(None, _call)
        return result.get("passed", False)

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            except Exception:
                pass

# ── voice call helper ─────────────────────────────────────────────────────────
async def voice_call(prompt: str, session_id: str | None = None) -> str:
    sid = session_id or str(uuid.uuid4())
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/voice",
            json={"session_id": sid, "prompt": prompt, "user_id": ADMIN_ID},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            return data.get("response", "")

# ── Eval Suite ────────────────────────────────────────────────────────────────
class EvalSuite:
    def __init__(self, judge: DockerJudge):
        self.judge = judge
        self.results = {
            "correctness": {},
            "performance": {"scenarios": {}},
            "raw":         {"rag": [], "tools": []},
        }
        self._progress = {
            "rag":    {"done": 0, "total": len(RAG_QA_PAIRS),   "passed": 0},
            "tools":  {"done": 0, "total": len(TOOL_TEST_CASES), "passed": 0},
            "latency":{"done": 0, "total": len(LATENCY_SCENARIOS)},
        }

    async def warmup(self):
        print("Warming up backend...")
        await sse_broadcast({"type": "status", "message": "Warming up backend..."})
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{BASE_URL}/warmup",
                                        timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    return await resp.json()
        except Exception as e:
            print(f"  [warmup] Warning: {e}")

    # ── RAG ──────────────────────────────────────────────────────────────────
    async def _eval_one_rag(self, pair: dict, idx: int) -> dict:
        await sse_broadcast({
            "type": "rag_start", "idx": idx,
            "question": pair["q"], "total": len(RAG_QA_PAIRS)
        })
        try:
            response = await voice_call(pair["q"])
            passed   = await self.judge.judge(pair["q"], pair["a"], response)
        except Exception as e:
            print(f"  [rag] Error on #{idx}: {e}")
            response, passed = "", False

        self._progress["rag"]["done"]   += 1
        self._progress["rag"]["passed"] += int(passed)
        item = {"question": pair["q"], "expected": pair["a"],
                "actual": response, "passed": passed, "doc_id": pair.get("doc_id", "")}
        await sse_broadcast({
            "type": "rag_result", "idx": idx, "passed": passed,
            "question": pair["q"], "expected": pair["a"], "actual": response,
            "progress": self._progress["rag"]
        })
        return item

    async def run_correctness_rag(self):
        print(f"\nRunning RAG Correctness ({len(RAG_QA_PAIRS)} questions)...")
        await sse_broadcast({"type": "phase", "phase": "rag",
                             "total": len(RAG_QA_PAIRS), "message": "RAG Correctness"})
        # Sequential to avoid hammering the LLM lock
        results = []
        for i, pair in enumerate(RAG_QA_PAIRS):
            r = await self._eval_one_rag(pair, i)
            results.append(r)
        passed    = sum(1 for r in results if r["passed"])
        precision = passed / len(results)
        self.results["correctness"]["rag_precision"] = precision
        self.results["correctness"]["rag_passed"]    = passed
        self.results["correctness"]["rag_total"]     = len(results)
        self.results["raw"]["rag"]                   = results
        print(f"  RAG Precision: {passed}/{len(results)} = {precision:.2%}")
        await sse_broadcast({"type": "phase_done", "phase": "rag",
                             "precision": precision, "passed": passed, "total": len(results)})

    # ── Tools ─────────────────────────────────────────────────────────────────
    async def _eval_one_tool(self, test: dict, idx: int) -> dict:
        await sse_broadcast({
            "type": "tool_start", "idx": idx,
            "prompt": test["prompt"], "expected_tool": test["expected_tool"],
            "total": len(TOOL_TEST_CASES)
        })
        try:
            response = await voice_call(test["prompt"])
            passed   = await self.judge.judge(
                question=test["prompt"],
                expected=f"Use the {test['expected_tool']} tool and return relevant results",
                actual=response,
            )
        except Exception as e:
            print(f"  [tool] Error on #{idx}: {e}")
            response, passed = "", False

        self._progress["tools"]["done"]   += 1
        self._progress["tools"]["passed"] += int(passed)
        item = {"prompt": test["prompt"], "expected_tool": test["expected_tool"],
                "actual": response, "passed": passed}
        await sse_broadcast({
            "type": "tool_result", "idx": idx, "passed": passed,
            "prompt": test["prompt"], "expected_tool": test["expected_tool"],
            "actual": response, "progress": self._progress["tools"]
        })
        return item

    async def run_correctness_tools(self):
        print(f"\nRunning Tool Correctness ({len(TOOL_TEST_CASES)} cases)...")
        await sse_broadcast({"type": "phase", "phase": "tools",
                             "total": len(TOOL_TEST_CASES), "message": "Tool Accuracy"})
        results  = []
        for i, test in enumerate(TOOL_TEST_CASES):
            r = await self._eval_one_tool(test, i)
            results.append(r)
        passed   = sum(1 for r in results if r["passed"])
        accuracy = passed / len(results)
        self.results["correctness"]["tool_accuracy"] = accuracy
        self.results["correctness"]["tool_passed"]   = passed
        self.results["correctness"]["tool_total"]    = len(results)
        self.results["raw"]["tools"]                 = results
        print(f"  Tool Accuracy: {passed}/{len(results)} = {accuracy:.2%}")
        await sse_broadcast({"type": "phase_done", "phase": "tools",
                             "accuracy": accuracy, "passed": passed, "total": len(results)})

    # ── Latency ───────────────────────────────────────────────────────────────
    async def measure_latency(self, name: str, prompt: str) -> dict:
        print(f"\nLatency '{name}' ({LATENCY_TRIALS} trials)...")
        await sse_broadcast({"type": "latency_start", "scenario": name, "trials": LATENCY_TRIALS})
        metrics = {"ttft": [], "itl": [], "e2e": []}

        for i in range(LATENCY_TRIALS):
            sid     = str(uuid.uuid4())
            t_start = time.perf_counter()
            t_first = None
            tokens  = []
            try:
                async with websockets.connect(
                    f"{WS_URL}?session_id={sid}", ping_interval=None, open_timeout=30
                ) as ws:
                    await ws.send(json.dumps({"session_id": sid, "message": prompt, "user_id": ADMIN_ID}))
                    async for msg in ws:
                        d = json.loads(msg)
                        if "token" in d:
                            if t_first is None:
                                t_first = time.perf_counter()
                            tokens.append(d["token"])
                        if d.get("done"):
                            break
            except Exception as e:
                print(f"  Trial {i+1} error: {e}")
                continue

            t_end = time.perf_counter()
            e2e   = (t_end - t_start) * 1000
            ttft  = (t_first - t_start) * 1000 if t_first else e2e
            itl   = (e2e - ttft) / max(1, len(tokens) - 1) if len(tokens) > 1 else 0
            metrics["ttft"].append(ttft)
            metrics["itl"].append(itl)
            metrics["e2e"].append(e2e)
            print(f"  [{i+1}/{LATENCY_TRIALS}] e2e={e2e:.0f}ms ttft={ttft:.0f}ms tokens={len(tokens)}")
            await sse_broadcast({"type": "latency_trial", "scenario": name, "trial": i+1,
                                 "e2e": e2e, "ttft": ttft, "tokens": len(tokens)})

        if not metrics["e2e"]:
            return {"mean":0,"median":0,"p90":0,"std":0,"ci_95":0,"ttft_avg":0,"itl_avg":0,"samples":[]}
        m     = self._summarize(metrics)
        self._progress["latency"]["done"] += 1
        await sse_broadcast({"type": "latency_done", "scenario": name, "metrics": m})
        return m

    def _summarize(self, m: dict) -> dict:
        std = statistics.stdev(m["e2e"]) if len(m["e2e"]) > 1 else 0
        return {
            "mean":     statistics.mean(m["e2e"]),
            "median":   statistics.median(m["e2e"]),
            "p90":      sorted(m["e2e"])[int(len(m["e2e"]) * 0.9)],
            "std":      std,
            "ci_95":    1.96 * (std / (len(m["e2e"]) ** 0.5)),
            "ttft_avg": statistics.mean(m["ttft"]),
            "itl_avg":  statistics.mean(m["itl"]),
            "samples":  m["e2e"],
        }

    # ── Throughput ────────────────────────────────────────────────────────────
    async def run_throughput(self):
        print("\nThroughput test (5 users × 3 turns)...")
        await sse_broadcast({"type": "phase", "phase": "throughput", "message": "Throughput test"})
        start = time.perf_counter()
        await asyncio.gather(*[self._user_session(i) for i in range(5)])
        dur = time.perf_counter() - start
        tps = (5 * 3) / dur
        self.results["performance"]["throughput"]      = tps
        self.results["performance"]["throughput_secs"] = round(dur, 2)
        print(f"  {tps:.2f} turns/sec ({dur:.1f}s)")
        await sse_broadcast({"type": "phase_done", "phase": "throughput", "tps": tps})

    async def _user_session(self, uid: int):
        sid = f"concurr_{uid}_{uuid.uuid4().hex[:6]}"
        for _ in range(3):
            try:
                await voice_call("Tell me about Daraz", session_id=sid)
            except Exception:
                pass

    # ── Save outputs ──────────────────────────────────────────────────────────
    async def save_json(self):
        out = {
            "generated_at": time.ctime(),
            "system":       await get_system_info(),
            "correctness":  self.results["correctness"],
            "performance":  self.results["performance"],
            "raw":          self.results["raw"],
        }
        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Saved → eval_results.json")

    async def save_report(self):
        cor  = self.results["correctness"]
        perf = self.results["performance"]
        sys  = await get_system_info()
        lines = [
            f"# NLP Assignment 5 - Evaluation Report",
            f"Generated on: {time.ctime()}\n",
            f"## 1. System Configuration",
            f"- **OS**: {sys['os']}",
            f"- **CPU**: {sys['processor']}",
            f"- **RAM**: {sys['ram']}",
            f"- **Python**: {sys['python_version']}",
            f"- **Judge**: Qwen 2.5-3B-Instruct (Docker, neutral system prompt)\n",
            f"## 2. Correctness Metrics",
            f"| Metric | Value | Detail |",
            f"|--------|-------|--------|",
            f"| RAG Precision@1 | {cor.get('rag_precision',0):.2%} | {cor.get('rag_passed',0)}/{cor.get('rag_total',0)} |",
            f"| Tool Accuracy   | {cor.get('tool_accuracy',0):.2%} | {cor.get('tool_passed',0)}/{cor.get('tool_total',0)} |\n",
            f"## 3. Latency (Single Turn, {LATENCY_TRIALS} trials each)",
            f"| Scenario | Mean E2E (ms) | 95% CI | Median (ms) | P90 (ms) | Avg TTFT (ms) |",
            f"|----------|---------------|--------|-------------|----------|---------------|",
        ]
        for name, m in perf.get("scenarios", {}).items():
            lines.append(f"| {name} | {m['mean']:.1f} | ±{m['ci_95']:.1f} | {m['median']:.1f} | {m['p90']:.1f} | {m['ttft_avg']:.1f} |")
        lines += [
            f"\n## 4. Throughput",
            f"- **Turns/sec**: {perf.get('throughput',0):.2f}",
            f"- **Duration**: {perf.get('throughput_secs',0):.1f}s",
        ]
        with open("eval_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Saved → eval_report.md")

    # ── Main ──────────────────────────────────────────────────────────────────
    async def run_all(self):
        print("=" * 60)
        print("  Daraz AI Evaluation Suite")
        print("=" * 60)
        await sse_broadcast({"type": "start", "message": "Eval suite started",
                             "rag_total": len(RAG_QA_PAIRS),
                             "tool_total": len(TOOL_TEST_CASES),
                             "latency_scenarios": [s["name"] for s in LATENCY_SCENARIOS]})
        await self.warmup()
        await self.run_correctness_rag()
        await self.run_correctness_tools()

        for s in LATENCY_SCENARIOS:
            m = await self.measure_latency(s["name"], s["prompt"])
            self.results["performance"]["scenarios"][s["name"]] = m

        await self.run_throughput()
        await self.save_json()
        await self.save_report()

        cor  = self.results["correctness"]
        perf = self.results["performance"]
        print("\n" + "=" * 60)
        print(f"  RAG Precision : {cor.get('rag_precision',0):.2%}")
        print(f"  Tool Accuracy : {cor.get('tool_accuracy',0):.2%}")
        print(f"  Throughput    : {perf.get('throughput',0):.2f} turns/sec")
        print("=" * 60)

        await sse_broadcast({
            "type": "complete",
            "rag_precision":  cor.get("rag_precision", 0),
            "tool_accuracy":  cor.get("tool_accuracy", 0),
            "throughput":     perf.get("throughput", 0),
        })


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    sse_runner = await start_sse_server()
    judge      = DockerJudge()
    await judge.start()

    suite = EvalSuite(judge)
    try:
        await suite.run_all()
    finally:
        judge.stop()
        await sse_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
