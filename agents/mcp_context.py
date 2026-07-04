"""
MongoDB MCP context provider — persistent background event loop architecture.

The MongoDB MCP server (mongodb-mcp-server) is proven to work; the challenge
was running it from Streamlit, which executes code in short-lived worker
threads. Spawning the MCP subprocess in a throwaway event loop per request
doesn't give the stdio initialize handshake time to complete.

This module fixes that by mirroring how VS Code runs MCP: ONE long-lived
event loop on a dedicated daemon thread, started once, that owns the MCP
client connection for the whole app lifetime. Each question is submitted to
that persistent loop via run_coroutine_threadsafe — we never create a new
loop or a new subprocess per query. The subprocess is launched once, lazily,
on the first question, and reused thereafter.

This also removes the per-query npx cold-start latency: the subprocess boots
once, then stays warm.

Augments retrieval in qa_agent; never replaces it. Any failure returns "" so
the QA agent proceeds on retrieval context alone. Set MCP_DISABLE=1 to skip.
"""

import os
import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt import create_react_agent
    from langchain_aws import ChatBedrock
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"
DB_NAME = "legacylens"


# ---------------------------------------------------------------------------
# Persistent background loop + MCP session (module-level singletons)
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_session_ctx = None          # the async context manager for the MCP session
_session = None              # the live MCP session
_tools = None                # loaded LangChain tools
_agent = None                # the ReAct agent (built once)
_init_lock = threading.Lock()
_init_done = False
_init_error: str | None = None


def _start_background_loop() -> None:
    """Start one daemon thread running a persistent event loop."""
    global _loop, _thread

    if _loop is not None:
        return

    def _run(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    # On Windows, subprocess support requires a Proactor loop.
    import sys
    if sys.platform == "win32":
        _loop = asyncio.ProactorEventLoop()
    else:
        _loop = asyncio.new_event_loop()

    _thread = threading.Thread(target=_run, args=(_loop,), daemon=True, name="mcp-loop")
    _thread.start()


async def _ainit_session() -> None:
    """
    Open the MCP session ONCE on the persistent loop, load tools, build the
    agent. Runs inside the background loop.
    """
    global _session_ctx, _session, _tools, _agent

    mongo_uri = os.environ["MONGO_URI"]
    client = MultiServerMCPClient(
        {
            "mongodb": {
                "transport": "stdio",
                "command":   "npx",
                "args": [
                    "-y", "mongodb-mcp-server",
                    "--connectionString", mongo_uri,
                ],
            }
        }
    )

    # Enter the session context manager and KEEP it open (don't exit it) so the
    # subprocess stays alive for the app's lifetime.
    _session_ctx = client.session("mongodb")
    _session = await _session_ctx.__aenter__()
    _tools = await load_mcp_tools(_session)

    model = ChatBedrock(
        model_id=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    _agent = create_react_agent(model, _tools)


def _ensure_initialised(init_timeout: float) -> bool:
    """
    Idempotently start the loop and initialise the MCP session. Returns True if
    the agent is ready, False if initialisation failed (so the caller skips MCP).
    """
    global _init_done, _init_error

    if _init_done:
        return _agent is not None

    with _init_lock:
        if _init_done:
            return _agent is not None

        try:
            _start_background_loop()
            # Run the async init on the background loop and wait for it
            fut = asyncio.run_coroutine_threadsafe(_ainit_session(), _loop)
            fut.result(timeout=init_timeout)
            _init_done = True
            print(f"[mcp_context] MCP session initialised ({len(_tools)} tools) — "
                  f"subprocess is warm and will be reused")
            return True
        except Exception as exc:
            _init_error = str(exc)
            _init_done = True   # don't retry every question if it's broken
            print(f"[mcp_context] MCP init failed ({exc}) — MCP disabled for this run")
            return False


# ---------------------------------------------------------------------------
# Per-question query (runs on the persistent loop)
# ---------------------------------------------------------------------------

async def _aquery(question: str, repo_id: str, mode: str = "aggregate") -> str:
    if mode == "retrieve":
        # PRIMARY retrieval mode: the agent is the main way this question gets
        # answered, so let it query freely and return whatever fits — full
        # lists, orderings, counts, specific entities. No aggregate-only limit.
        instruction = (
            f"You are a MongoDB analyst for a codebase-analysis tool. "
            f"The database is '{DB_NAME}'. Collections:\n"
            f"  - entities (entity_id, type, file_path, line_start, line_end, "
            f"risk_score, risk_band, fan_in, fan_out, has_tests, repo_id, "
            f"summary_purpose)\n"
            f"  - edges (from_id, to_id, edge_type) — from_id CALLS to_id\n\n"
            f"All documents for the current repo have repo_id = '{repo_id}'. "
            f"ALWAYS filter by this repo_id.\n\n"
            f"The user asked: \"{question}\"\n\n"
            f"This is the PRIMARY data source for answering. Run whatever "
            f"MongoDB queries are needed (find, count, aggregate, sort) to fully "
            f"answer it. If they want a list or ordering of entities, fetch ALL "
            f"matching entities (not a sample) with the fields needed and return "
            f"them in the right order. If they want counts or breakdowns, "
            f"aggregate. Return the actual query RESULTS as clear plain text — "
            f"include entity_id and file_path for entities so they can be cited. "
            f"Do not truncate a requested list. Be complete but concise."
        )
    else:
        # AGGREGATE mode: supplementary context alongside a deterministic path.
        instruction = (
            f"You are a MongoDB analyst for a codebase-analysis tool. "
            f"The database is '{DB_NAME}'. The relevant collections are:\n"
            f"  - entities (fields: entity_id, type, file_path, risk_score, "
            f"risk_band, fan_in, fan_out, has_tests, repo_id)\n"
            f"  - edges (fields: from_id, to_id, edge_type)\n\n"
            f"All documents for the current repo have repo_id = '{repo_id}'. "
            f"Always filter by this repo_id.\n\n"
            f"The user asked: \"{question}\"\n\n"
            f"Run at most TWO MongoDB queries (counts, aggregations, or finds) that "
            f"gather useful AGGREGATE or STRUCTURAL facts to help answer this — for "
            f"example total counts, breakdowns by file, counts of untested entities, "
            f"or the highest-fan-in entities. Do NOT answer the question yourself. "
            f"Return the factual query results concisely as plain text. Under 150 words."
        )
    result = await _agent.ainvoke({"messages": [{"role": "user", "content": instruction}]})
    messages = result.get("messages", [])

    # Extract the final answer text
    text = ""
    if messages:
        final = messages[-1]
        content = getattr(final, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        text = content.strip()

    # Extract the tool activity (which MCP tools ran, with what args, and what
    # they returned) by walking the message history. AIMessages carry
    # tool_calls; ToolMessages carry the corresponding results.
    tool_calls = []
    pending = {}   # tool_call_id -> {name, args} awaiting its result
    for m in messages:
        # Tool invocations live on AI messages
        calls = getattr(m, "tool_calls", None) or []
        for c in calls:
            cid  = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
            name = c.get("name") if isinstance(c, dict) else getattr(c, "name", "")
            args = c.get("args") if isinstance(c, dict) else getattr(c, "args", {})
            pending[cid] = {"name": name, "args": args}
        # Tool results are ToolMessages (type == "tool")
        if getattr(m, "type", None) == "tool" or m.__class__.__name__ == "ToolMessage":
            cid = getattr(m, "tool_call_id", None)
            rc  = getattr(m, "content", "") or ""
            if isinstance(rc, list):
                rc = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in rc
                )
            info = pending.pop(cid, {"name": getattr(m, "name", "tool"), "args": {}})
            tool_calls.append({
                "name":   info["name"],
                "args":   info["args"],
                "result": rc.strip()[:1500],   # cap result size for the UI
            })

    return {"text": text, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Public sync entry point used by qa_agent
# ---------------------------------------------------------------------------

def get_mcp_context(question: str, repo_id: str, mode: str = "aggregate") -> dict:
    """
    Gather MongoDB context for a question via the persistent MCP session.

    mode="aggregate" — supplementary facts (counts/breakdowns), used alongside
                       a deterministic retrieval path. Concise.
    mode="retrieve"  — PRIMARY retrieval: the agent queries freely and returns
                       full results (lists, orderings, counts) for open-ended
                       or whole-codebase questions.

    Returns a dict: {"text": <str>, "tool_calls": [{"name","args","result"}, ...]}.
    On any failure / if disabled, returns {"text": "", "tool_calls": []} so the
    caller can always rely on the shape.

    First call initialises the session (boots the subprocess once); later calls
    reuse it.
    """
    empty = {"text": "", "tool_calls": []}

    if not _MCP_AVAILABLE:
        print("[mcp_context] langchain-mcp-adapters not installed — skipping")
        return empty

    if os.environ.get("MCP_DISABLE", "0") == "1":
        print("[mcp_context] MCP_DISABLE=1 — skipping")
        return empty

    init_timeout  = float(os.environ.get("MCP_INIT_TIMEOUT_SECONDS", "120"))
    query_timeout = float(os.environ.get("MCP_QUERY_TIMEOUT_SECONDS", "45"))

    if not _ensure_initialised(init_timeout):
        return empty

    try:
        fut = asyncio.run_coroutine_threadsafe(_aquery(question, repo_id, mode), _loop)
        return fut.result(timeout=query_timeout)
    except FutureTimeoutError:
        print(f"[mcp_context] MCP query timed out after {query_timeout}s — continuing without it")
        return empty
    except Exception as exc:
        print(f"[mcp_context] MCP query failed ({exc}) — continuing without it")
        return empty