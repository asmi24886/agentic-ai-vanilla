"""
MULTI-AGENT STOCK DEMO  --  PRIVATE PER-AGENT HISTORY, SUPERVISOR SEES ONLY SUMMARIES
(pure Python, no framework, human-in-the-loop included)

WHAT CHANGED vs the "shared history" version
  BEFORE:  one list state["messages"] that every agent read and wrote.
  NOW:     one list PER AGENT: state["histories"] = {"supervisor": [...], "researcher": [...], "buyer": [...]}
           Each agent is only ever shown its OWN list.

THE TRICK THAT MAKES THE SUPERVISOR SEE ONLY SUMMARIES
  When the supervisor calls transfer_to_researcher(task=...), we leave that tool call OPEN
  (no result yet). The researcher then works in its own private history. When it finishes with
  handoff_back(summary=...), that summary becomes the RESULT of the supervisor's open call.

      supervisor's history           researcher's history (private)
      ────────────────────           ──────────────────────────────
      user: research + buy Apple
      call: transfer_to_researcher ──► user: "Task from supervisor: ..."   (fresh, empty start)
        (call stays OPEN)              call: ask_user("which ticker?")
                                       tool: "AAPL"
                                       call: get_stock_price ... get_news ...
                                       call: handoff_back(summary) ──┐
      tool result: <the summary>  ◄───────────────────────────────────┘
      call: transfer_to_buyer ──────► (buyer's own private history starts the same way)

  To the supervisor it looks like a normal tool call that took a while and returned a short answer.
  Analogy: a manager hands a task to a specialist. She doesn't sit in on their phone calls;
  she only reads the report they hand back.

TRADE-OFFS
  + Cheaper: each Claude call sends a short, focused history instead of everything.
  + Focused: the buyer isn't distracted by the researcher's tool noise.
  + Isolated: easier to test and reason about each agent alone.
  - Lossy BY DESIGN: a sub-agent only knows what the supervisor put in the 'task' text.
    If the researcher's summary forgets the ticker, the buyer can't know it. Summary quality matters.

WHAT WE SEND TO CLAUDE ON EACH CALL
     system prompt -> PER AGENT
     tools         -> PER AGENT
     messages      -> PER AGENT: only THAT agent's private history (NOT everyone's)
  Each spot is marked  >>> SENDING TO CLAUDE <<<
"""
import json, os, re, uuid
from dataclasses import dataclass, field
from typing import Callable


# ══════════════════════════════════════════════════════════════════════════
# 1. TOOLS -- what agents can actually DO (prints stand in for real API calls)
# ══════════════════════════════════════════════════════════════════════════
def get_stock_price(ticker):
    print(f"   [API] fetching price for {ticker}")
    return f"{ticker} trades at $190.12"

def get_news(ticker):
    print(f"   [API] fetching news for {ticker}")
    return "Headlines: strong iPhone demand, analyst upgrade to Buy"

def place_order(ticker, qty, order_type):
    print(f"   [API] ORDER PLACED -> BUY {qty} {ticker} ({order_type})")   # the dangerous side effect
    return f"Order ORD-1001 filled: BUY {qty} {ticker} ({order_type})"


@dataclass
class Tool:
    name: str
    desc: str                 # the LLM reads this to decide when to use the tool
    params: dict              # {argument name: description}
    fn: Callable = None       # None => "control tool" handled by the engine itself
    risky: bool = False       # True => engine pauses for human approval BEFORE running it

ASK  = Tool("ask_user", "Ask the user a clarifying question and wait for the answer.",
            {"question": "The question to ask"})
BACK = Tool("handoff_back", "Finished or blocked: return control to the supervisor.",
            {"summary": "Everything the supervisor needs to know. Include key facts (e.g. the ticker)."})


# ══════════════════════════════════════════════════════════════════════════
# 2. AGENTS -- a name + instructions + allowed tools
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Agent:
    name: str
    prompt: str
    tools: list = field(default_factory=list)

ONE_TOOL = " Call exactly one tool at a time."

AGENTS = {a.name: a for a in [

    Agent("supervisor",
          "You route work. Delegate research to 'researcher', trades to 'buyer'. "
          "Sub-agents cannot see this conversation, so put every detail they need in the 'task' text. "
          "Never research or trade yourself. When all work is done, reply to the user." + ONE_TOOL,
          [Tool("transfer_to_researcher", "Hand off to the researcher.", {"task": "Full instructions"}),
           Tool("transfer_to_buyer",      "Hand off to the buyer.",      {"task": "Full instructions incl. ticker"})]),

    Agent("researcher",
          "You research stocks. If the ticker is unclear, ask_user. "
          "When done, handoff_back with a summary that includes the ticker." + ONE_TOOL,
          [Tool("get_stock_price", "Current price.", {"ticker": "Ticker"}, get_stock_price),
           Tool("get_news",        "Latest news.",   {"ticker": "Ticker"}, get_news),
           ASK, BACK]),

    Agent("buyer",
          "You place buy orders. The task text tells you what to buy. Collect quantity and market/limit "
          "via ask_user, then call place_order. The system will ask the human to approve it. "
          "If the result says REJECTED, handoff_back. If it says NOT EXECUTED with a requested change, "
          "call place_order again with the revised details." + ONE_TOOL,
          [Tool("place_order", "Place a buy order (needs human approval).",
                {"ticker": "Ticker", "qty": "Shares", "order_type": "market or limit"},
                lambda ticker, qty, order_type: place_order(ticker, int(qty), order_type),
                risky=True),                                  # <-- THE HITL SWITCH
           ASK, BACK]),
]}


# ══════════════════════════════════════════════════════════════════════════
# 3. THE STORE -- the only memory in the system (Redis/Postgres in production)
# ══════════════════════════════════════════════════════════════════════════
def new_state():
    return {
        "version": 0,            # bumped per save; detects two requests colliding on one thread
        "active": "supervisor",  # WHO is in charge. A handoff is just changing this.
        "pending": None,         # None, or a paused action waiting for the human:
                                 #   {"kind": "question" | "approval", "call": <the tool call>}
        "handoff": None,         # the supervisor's OPEN transfer call, waiting for a summary:
                                 #   {"id": <call id>, "name": "transfer_to_buyer"}
        "histories": {           # <-- NEW: one private message list per agent
            "supervisor": [],
            "researcher": [],
            "buyer": [],
        },
    }

class Store:
    def __init__(self):
        self.db = {}             # thread_id -> JSON string

    def load(self, tid):
        return json.loads(self.db[tid]) if tid in self.db else new_state()

    def save(self, tid, state):
        current = json.loads(self.db[tid])["version"] if tid in self.db else 0
        if current != state["version"]:                  # optimistic lock
            raise RuntimeError("concurrent update, retry")
        state["version"] += 1
        self.db[tid] = json.dumps(state)


# ══════════════════════════════════════════════════════════════════════════
# 4. THE ENGINE -- run_turn(): one call == one HTTP request
# ══════════════════════════════════════════════════════════════════════════
def tool_msg(call_id, name, content):
    """A 'tool result' message. Must reuse the call's id so the LLM API accepts the history."""
    return {"role": "tool", "id": call_id, "name": name, "content": content}


def run_tool(agent, name, args):
    """Run a normal tool, looking it up ONLY in this agent's own tool list."""
    tool = {t.name: t for t in agent.tools}.get(name)
    return tool.fn(**args) if tool else f"Unknown tool {name}"


def classify(text):
    """Turn the human's reply to an approval prompt into: approve / reject / feedback.
    (A real UI would use buttons and return a structured answer.)"""
    t = text.strip().lower()
    if t in {"yes", "y", "approve", "ok", "confirm", "go"}: return "approve"
    if t in {"no", "n", "reject", "cancel", "stop"}:        return "reject"
    return "feedback"


def run_turn(tid, user_text, store, brain, max_steps=20):
    # ── STEP 1: LOAD THE TICKET (server remembers nothing between requests) ──
    state = store.load(tid)

    # ── STEP 2: FILE THE USER'S MESSAGE, into the ACTIVE agent's private history ──
    # Whoever is active is who the user is "talking to" right now, so only their history grows.
    active_history = state["histories"][state["active"]]
    pending = state["pending"]

    if pending:
        state["pending"] = None                  # resuming, so clear the pause marker
        call = pending["call"]                   # the tool call that was paused

        if pending["kind"] == "question":
            # RESUME after a clarifying question: the user's text IS the ask_user result.
            active_history.append(tool_msg(call["id"], "ask_user", user_text))

        else:  # "approval"
            # RESUME after a risky tool call was intercepted (tool has NOT run yet).
            agent = AGENTS[state["active"]]
            decision = classify(user_text)
            if decision == "approve":
                # Run exactly the args the human saw, not anything the LLM re-generates.
                result = run_tool(agent, call["name"], call["args"])
            elif decision == "reject":
                result = "REJECTED: the user cancelled this action. Do not retry it."
            else:
                result = (f"NOT EXECUTED: the user wants a change instead: \"{user_text}\". "
                          f"Revise and try again, or ask them.")
            active_history.append(tool_msg(call["id"], call["name"], result))
    else:
        active_history.append({"role": "user", "content": user_text})

    # ── STEP 3: THE LOOP. Each pass = ask the active agent's LLM what's next, then do it. ──
    # Ends (returns to the user) when: an agent speaks plain text, asks a question,
    # or a risky tool needs approval. Handoffs do NOT end it; they continue the loop.
    for _ in range(max_steps):

        # 3a. Who's in charge? Fetch THEIR prompt, THEIR tools, and THEIR private history.
        agent = AGENTS[state["active"]]
        history = state["histories"][agent.name]

        # 3b. >>> SENDING TO CLAUDE <<<  (inside brain())
        #     messages = `history` -> ONLY this agent's private list. The other agents' lists
        #                are never sent. system and tools are also this agent's only.
        reply = brain(agent, history)
        calls = reply.get("tool_calls", [])[:1]

        # 3c. Record the model's reply in THIS agent's history only.
        history.append({"role": "assistant", "content": reply.get("text", ""), "tool_calls": calls})

        # 3d. Plain text, no tool: the agent is speaking to the user. Save and end the turn.
        if not calls:
            store.save(tid, state)
            return reply["text"]

        call = calls[0]
        name, args = call["name"], call["args"]

        # 3e. PAUSE TYPE 1 -- agent asked a clarifying question.
        if name == "ask_user":
            state["pending"] = {"kind": "question", "call": call}
            store.save(tid, state)
            return f"[{agent.name}] {args['question']}"

        # 3f. HANDOFF supervisor -> sub-agent.
        if name.startswith("transfer_to_"):
            target = name.removeprefix("transfer_to_")
            # Remember the supervisor's call id. We deliberately give it NO result yet:
            # the sub-agent's summary will become its result when the sub-agent hands back.
            state["handoff"] = {"id": call["id"], "name": name}
            # The sub-agent starts with a FRESH private history containing only the task text.
            # (This is the ONLY context it gets from the supervisor.)
            state["histories"][target] = [{"role": "user", "content": f"Task from supervisor: {args['task']}"}]
            state["active"] = target
            print(f"   -- handoff: {agent.name} -> {target}")
            continue                                        # loop: the sub-agent now acts

        # 3g. HANDOFF sub-agent -> supervisor.
        if name == "handoff_back":
            # Tidy the sub-agent's own history so it stays valid (kept for audit/debugging,
            # but never shown to the supervisor).
            history.append(tool_msg(call["id"], name, "Handed back to supervisor."))
            # THE KEY LINE: the summary becomes the RESULT of the supervisor's open transfer call.
            # This is the only thing the supervisor will ever see of the sub-agent's work.
            h = state["handoff"]
            state["histories"]["supervisor"].append(tool_msg(h["id"], h["name"], args["summary"]))
            state["handoff"] = None
            state["active"] = "supervisor"
            print(f"   -- handoff: {agent.name} -> supervisor")
            continue

        # 3h. A normal tool (price lookup, news, place_order...).
        tool = {t.name: t for t in agent.tools}.get(name)

        # PAUSE TYPE 2 -- risky tool: DO NOT RUN IT. Park the call, ask the human.
        # This check is in code, so even a misbehaving LLM cannot trade without a yes.
        if tool and tool.risky:
            state["pending"] = {"kind": "approval", "call": call}
            store.save(tid, state)
            return (f"⚠️  [{agent.name}] wants to run {name}({args}).\n"
                    f"    Reply 'yes' to approve, 'no' to cancel, "
                    f"or type a change (e.g. 'make it 5 shares').")

        # 3i. Safe tool: run it and file the result in THIS agent's history, then loop.
        history.append(tool_msg(call["id"], name, run_tool(agent, name, args)))

    store.save(tid, state)
    return "Stopped: too many steps."


# ══════════════════════════════════════════════════════════════════════════
# 5. BRAINS -- the only code that touches an LLM.
#    A brain is any function: (agent, history) -> {"text": ..., "tool_calls": [...]}
#    NOTE: `msgs` below is now ONE agent's private history, not a shared one.
# ══════════════════════════════════════════════════════════════════════════
def call(name, **args):
    """Build a 'please call this tool' reply, the way an LLM would."""
    return {"tool_calls": [{"id": uuid.uuid4().hex[:8], "name": name, "args": args}]}

def result_of(msgs, tool):    # latest RESULT of a tool, within this agent's own history
    return next(m["content"] for m in reversed(msgs) if m["role"] == "tool" and m["name"] == tool)
def args_of(msgs, tool):      # latest ARGUMENTS a tool was called with, within this agent's own history
    return next(c["args"] for m in reversed(msgs) for c in m.get("tool_calls", []) if c["name"] == tool)


def scripted_brain(agent, msgs):
    """
    FAKE LLM so the demo runs offline. It can only see `msgs` = the ACTIVE AGENT'S PRIVATE history,
    exactly like a real model would. Notice the buyer must dig the ticker out of its TASK TEXT,
    because it cannot see the researcher's tool calls.
    """
    last = msgs[-1]

    # ---- SUPERVISOR (its history: user msg, its transfer calls, and the SUMMARIES as results) ----
    if agent.name == "supervisor":
        if last["role"] == "user":
            return call("transfer_to_researcher", task=last["content"])
        if last["name"] == "transfer_to_researcher":     # researcher's SUMMARY arrived -> now buy
            return call("transfer_to_buyer",
                        task=f"Buy the stock if research looks good. Research summary: {last['content']}")
        return {"text": f"Summary: {last['content']}"}   # buyer's summary arrived -> tell the user

    # ---- RESEARCHER (its history starts with the task from the supervisor) ----
    if agent.name == "researcher":
        if last["role"] == "user":                       # just got the task, ticker unknown -> ask
            return call("ask_user", question="Which ticker symbol do you mean?")
        fn = last["name"]
        if fn == "ask_user":                             # got "AAPL" -> price
            return call("get_stock_price", ticker=last["content"].strip().upper())
        if fn == "get_stock_price":
            return call("get_news", ticker=args_of(msgs, "get_stock_price")["ticker"])
        if fn == "get_news":                             # done -> summary (includes the ticker!)
            return call("handoff_back",
                        summary=f"{result_of(msgs,'get_stock_price')}. {last['content']}. Looks bullish.")

    # ---- BUYER (its history starts with the task, which carries the research summary) ----
    if agent.name == "buyer":
        # Ticker comes from the TASK TEXT: the first ALL-CAPS word, e.g. "AAPL".
        tkr = re.search(r"\b[A-Z]{2,5}\b", msgs[0]["content"]).group()
        if last["role"] == "user":                       # just got the task -> collect details
            return call("ask_user", question="How many shares, and market or limit order?")
        fn = last["name"]
        if fn == "ask_user":                             # got details -> straight to place_order
            qty = re.search(r"\d+", last["content"]).group()          # (engine will intercept it!)
            otype = "limit" if "limit" in last["content"].lower() else "market"
            return call("place_order", ticker=tkr, qty=qty, order_type=otype)
        if fn == "place_order":                          # the engine handed us the outcome:
            c = last["content"]
            if c.startswith("NOT EXECUTED"):             #   user asked for a change -> try again
                prev = args_of(msgs, "place_order")
                qty = re.search(r"\d+", c).group()
                return call("place_order", ticker=prev["ticker"], qty=qty, order_type=prev["order_type"])
            if c.startswith("REJECTED"):                 #   user said no -> report back
                return call("handoff_back", summary="User declined. No order placed.")
            return call("handoff_back", summary=c)       #   approved & filled -> report back


class ClaudeBrain:
    """
    The REAL LLM version (sketch, not run in this demo). Converts our simple message format
    to the Anthropic API format.
    """
    def __init__(self, model="claude-sonnet-5-5"):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def __call__(self, agent, msgs):
        # `msgs` is ONLY this agent's private history (the engine passed state["histories"][agent.name]).
        # Every private history starts with a user message and alternates properly, which the API requires.
        out = []
        for m in msgs:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = ([{"type": "text", "text": m["content"]}] if m["content"] else []) + \
                         [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                          for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks})
            else:   # tool results travel back to Claude as a "user" block
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}]})

        tools = [{"name": t.name, "description": t.desc,
                  "input_schema": {"type": "object", "required": list(t.params),
                                   "properties": {k: {"type": "string", "description": v}
                                                  for k, v in t.params.items()}}}
                 for t in agent.tools]

        # >>> SENDING TO CLAUDE <<<
        #   system   = this agent's prompt                      (PER AGENT)
        #   tools    = this agent's tools                       (PER AGENT)
        #   messages = `out` = this agent's PRIVATE history only (PER AGENT; others' work is invisible).
        #              For the supervisor that means: user msgs, its transfer calls, and the sub-agents'
        #              SUMMARIES. For a sub-agent: the task text plus its own tool calls and results.
        r = self.client.messages.create(model=self.model, max_tokens=1024,
                                        system=agent.prompt, messages=out, tools=tools)

        return {"text": "".join(b.text for b in r.content if b.type == "text"),
                "tool_calls": [{"id": b.id, "name": b.name, "args": b.input}
                               for b in r.content if b.type == "tool_use"]}


# ══════════════════════════════════════════════════════════════════════════
# 6. DEMO
# ══════════════════════════════════════════════════════════════════════════
def chat(title, tid, user_messages, store, brain):
    print(f"\n{'═'*70}\n{title}\n{'═'*70}")
    for text in user_messages:
        print(f"\nUSER: {text}")
        print("BOT : ", run_turn(tid, text, store, brain))

def dump(store, tid):
    """Print each agent's saved private history so you can SEE the separation."""
    print(f"\n{'─'*70}\nSAVED PRIVATE HISTORIES for {tid}\n{'─'*70}")
    for name, hist in store.load(tid)["histories"].items():
        print(f"\n  {name.upper()}  ({len(hist)} messages)")
        for m in hist:
            if m["role"] == "user":
                line = f"user: {m['content']}"
            elif m["role"] == "assistant":
                calls = " ".join(f"CALL {c['name']}({c['args']})" for c in m["tool_calls"])
                line = f"assistant: {m['content']} {calls}".strip()
            else:
                line = f"tool-result[{m['name']}]: {m['content']}"
            print(f"     {line[:105]}{'…' if len(line) > 105 else ''}")

if __name__ == "__main__":
    brain = ClaudeBrain() if os.getenv("ANTHROPIC_API_KEY") else scripted_brain
    store = Store()                                          # the ONLY thing shared between requests

    # Scenario A: change the order, then approve.
    chat("A) change the order, then approve", "thread-A", [
        "Research Apple and buy some if it looks good",
        "AAPL",
        "10 shares, market order",        # -> approval prompt
        "make it 5 shares instead",       # -> feedback: buyer revises, new approval prompt
        "yes",                            # -> place_order actually runs
    ], store, brain)
    dump(store, "thread-A")               # <- look at how little the supervisor's list contains

    # Scenario B: reject.
    chat("B) reject the order", "thread-B", [
        "Research Apple and buy some if it looks good",
        "AAPL",
        "10 shares, market order",
        "no",
    ], store, brain)
