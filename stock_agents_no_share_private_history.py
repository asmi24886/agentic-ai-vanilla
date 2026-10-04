"""
MULTI-AGENT STOCK DEMO  --  EVERY AGENT KEEPS ITS OWN PERSISTENT PRIVATE HISTORY
(pure Python, no framework, human-in-the-loop included)

THE THREE HISTORY PATTERNS (so you can place this file)
  1. SHARED history      one list everyone reads and writes.            -> stock_agents_hitl.py
  2. PRIVATE + RESET     each agent has its own list, but a sub-agent's list is wiped and restarted
                         on every handoff.                              -> stock_agents_private_history.py
  3. PRIVATE + PERSISTENT (THIS FILE)
                         each agent has its own list AND keeps it forever. When the supervisor hands
                         work to the buyer a second time, the buyer still remembers its first visit.

  Analogy: pattern 2 is a contractor who gets a blank sheet of paper every time you call.
           Pattern 3 is a contractor with a DIARY: each call adds a new entry, and they can flip
           back to what they did last time.

WHAT THE ONE-LINE DIFFERENCE IS
  Pattern 2 (handoff):   state["histories"][target] = [ task ]        # wipe + start over
  Pattern 3 (handoff):   state["histories"][target].append( task )    # add to the diary
  Everything else (open transfer call, summary as its result, HITL pauses) is unchanged.

WHAT THE SUPERVISOR SEES: still only its own diary = user messages, its transfer calls, and the
sub-agents' summaries (the summary becomes the result of the supervisor's open transfer call).

WHY THIS MATTERS (see the demo): after buying 5 AAPL, the user says "buy 3 more of the same".
The supervisor has no idea what "the same" means in detail, and it doesn't need to. It forwards the
words to the buyer, and the buyer finds the answer in ITS OWN diary (AAPL, market order).

TRADE-OFFS
  + Follow-ups just work: agents remember their own past work.
  - Diaries grow forever. Real systems COMPACT them: summarise old entries, drop stale tool output,
    or cap by tokens. (Marked with  "# COMPACT HERE?"  in the engine.)
  - A sub-agent may act on STALE memory ("the same" when the user meant something else).

WHAT WE SEND TO CLAUDE ON EACH CALL
     system prompt -> PER AGENT
     tools         -> PER AGENT
     messages      -> PER AGENT and PERSISTENT: that agent's whole diary so far, across ALL its visits.
                      (Never anyone else's.)
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
          "Sub-agents cannot see this conversation, but they DO remember their own earlier work, "
          "so for follow-ups you can pass the user's words along. "
          "Never research or trade yourself. When all work is done, reply to the user." + ONE_TOOL,
          [Tool("transfer_to_researcher", "Hand off to the researcher.", {"task": "Full instructions"}),
           Tool("transfer_to_buyer",      "Hand off to the buyer.",      {"task": "Full instructions"})]),

    Agent("researcher",
          "You research stocks. If the ticker is unclear, ask_user. "
          "When done, handoff_back with a summary that includes the ticker." + ONE_TOOL,
          [Tool("get_stock_price", "Current price.", {"ticker": "Ticker"}, get_stock_price),
           Tool("get_news",        "Latest news.",   {"ticker": "Ticker"}, get_news),
           ASK, BACK]),

    Agent("buyer",
          "You place buy orders. Each task arrives as a new entry in your diary; use your earlier "
          "orders to resolve phrases like 'the same'. For a new purchase, collect quantity and "
          "market/limit via ask_user, then call place_order. The system will ask the human to approve it. "
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
        "histories": {           # one PERSISTENT private diary per agent
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

    # ── STEP 2: FILE THE USER'S MESSAGE, into the ACTIVE agent's diary ──
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
    for _ in range(max_steps):

        # 3a. Who's in charge? Fetch THEIR prompt, THEIR tools, and THEIR diary.
        agent = AGENTS[state["active"]]
        history = state["histories"][agent.name]

        # COMPACT HERE?  Diaries only grow. In production, if `history` is too long, summarise the
        # oldest entries (keeping tool_use/tool_result pairs together) before sending it.

        # 3b. >>> SENDING TO CLAUDE <<<  (inside brain())
        #     messages = `history` -> this agent's WHOLE diary (all its visits), nobody else's.
        reply = brain(agent, history)
        calls = reply.get("tool_calls", [])[:1]

        # 3c. Record the model's reply in THIS agent's diary only.
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
            # Keep the supervisor's call OPEN: the sub-agent's summary will become its result.
            state["handoff"] = {"id": call["id"], "name": name}
            # *** THE ONE-LINE DIFFERENCE vs the reset version ***
            # APPEND the task to the sub-agent's existing diary instead of replacing it.
            # (First ever visit: the diary is empty, so this behaves exactly like before.)
            state["histories"][target].append(
                {"role": "user", "content": f"Task from supervisor: {args['task']}"})
            state["active"] = target
            print(f"   -- handoff: {agent.name} -> {target}")
            continue

        # 3g. HANDOFF sub-agent -> supervisor.
        if name == "handoff_back":
            # Close the sub-agent's own tool call so its diary stays valid for its NEXT visit.
            history.append(tool_msg(call["id"], name, "Handed back to supervisor."))
            # The summary becomes the RESULT of the supervisor's open transfer call.
            h = state["handoff"]
            state["histories"]["supervisor"].append(tool_msg(h["id"], h["name"], args["summary"]))
            state["handoff"] = None
            state["active"] = "supervisor"
            print(f"   -- handoff: {agent.name} -> supervisor")
            continue

        # 3h. A normal tool (price lookup, news, place_order...).
        tool = {t.name: t for t in agent.tools}.get(name)

        # PAUSE TYPE 2 -- risky tool: DO NOT RUN IT. Park the call, ask the human.
        if tool and tool.risky:
            state["pending"] = {"kind": "approval", "call": call}
            store.save(tid, state)
            return (f"⚠️  [{agent.name}] wants to run {name}({args}).\n"
                    f"    Reply 'yes' to approve, 'no' to cancel, "
                    f"or type a change (e.g. 'make it 5 shares').")

        # 3i. Safe tool: run it and file the result in THIS agent's diary, then loop.
        history.append(tool_msg(call["id"], name, run_tool(agent, name, args)))

    store.save(tid, state)
    return "Stopped: too many steps."


# ══════════════════════════════════════════════════════════════════════════
# 5. BRAINS -- the only code that touches an LLM.
#    A brain is any function: (agent, diary) -> {"text": ..., "tool_calls": [...]}
# ══════════════════════════════════════════════════════════════════════════
def call(name, **args):
    """Build a 'please call this tool' reply, the way an LLM would."""
    return {"tool_calls": [{"id": uuid.uuid4().hex[:8], "name": name, "args": args}]}

def result_of(msgs, tool):    # latest RESULT of a tool, within this agent's own diary
    return next(m["content"] for m in reversed(msgs) if m["role"] == "tool" and m["name"] == tool)
def args_of(msgs, tool):      # latest ARGUMENTS a tool was called with, within this agent's own diary
    return next(c["args"] for m in reversed(msgs) for c in m.get("tool_calls", []) if c["name"] == tool)
def ticker_from_tasks(msgs):  # newest task text that mentions an ALL-CAPS ticker, e.g. "AAPL"
    for m in reversed(msgs):
        if m["role"] == "user" and (t := re.search(r"\b[A-Z]{2,5}\b", m["content"])):
            return t.group()


def scripted_brain(agent, msgs):
    """
    FAKE LLM so the demo runs offline. It only sees `msgs` = the ACTIVE AGENT'S OWN DIARY.
    A real model would make these same choices by reading the prompt and the diary.
    """
    last = msgs[-1]

    # ---- SUPERVISOR (user msgs, its transfer calls, and the SUMMARIES as results) ----
    if agent.name == "supervisor":
        if last["role"] == "user":
            if "more" in last["content"].lower():        # follow-up like "buy 3 more of the same"
                return call("transfer_to_buyer", task=last["content"])   # forward the words as-is
            return call("transfer_to_researcher", task=last["content"])
        if last["name"] == "transfer_to_researcher":     # researcher's SUMMARY arrived -> now buy
            return call("transfer_to_buyer",
                        task=f"Buy the stock if research looks good. Research summary: {last['content']}")
        return {"text": f"Summary: {last['content']}"}   # buyer's summary arrived -> tell the user

    # ---- RESEARCHER ----
    if agent.name == "researcher":
        if last["role"] == "user":                       # got the task, ticker unknown -> ask
            return call("ask_user", question="Which ticker symbol do you mean?")
        fn = last["name"]
        if fn == "ask_user":                             # got "AAPL" -> price
            return call("get_stock_price", ticker=last["content"].strip().upper())
        if fn == "get_stock_price":
            return call("get_news", ticker=args_of(msgs, "get_stock_price")["ticker"])
        if fn == "get_news":                             # done -> summary (includes the ticker!)
            return call("handoff_back",
                        summary=f"{result_of(msgs,'get_stock_price')}. {last['content']}. Looks bullish.")

    # ---- BUYER (its diary can span several visits) ----
    if agent.name == "buyer":
        if last["role"] == "user":                       # a NEW task entry just arrived
            if re.search(r"\b[A-Z]{2,5}\b", last["content"]):   # names a ticker -> brand-new purchase
                return call("ask_user", question="How many shares, and market or limit order?")
            # No ticker in the task, e.g. "Buy 3 more of the same": resolve "the same" from MY OWN
            # earlier order in this diary. A wiped (reset) history could not do this.
            prev = args_of(msgs, "place_order")
            qty = re.search(r"\d+", last["content"]).group()
            return call("place_order", ticker=prev["ticker"], qty=qty, order_type=prev["order_type"])
        fn = last["name"]
        if fn == "ask_user":                             # got details -> straight to place_order
            qty = re.search(r"\d+", last["content"]).group()          # (engine will intercept it!)
            otype = "limit" if "limit" in last["content"].lower() else "market"
            return call("place_order", ticker=ticker_from_tasks(msgs), qty=qty, order_type=otype)
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
        # `msgs` is ONLY this agent's diary (state["histories"][agent.name]), spanning all its visits.
        out = []

        def push(role, content):
            # A returning sub-agent's diary has a tool result followed directly by a new task, i.e.
            # two "user" turns in a row. The API wants alternating roles, so merge neighbours.
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            if out and out[-1]["role"] == role:
                out[-1]["content"] = out[-1]["content"] + blocks
            else:
                out.append({"role": role, "content": blocks})

        for m in msgs:
            if m["role"] == "user":
                push("user", m["content"])
            elif m["role"] == "assistant":
                blocks = ([{"type": "text", "text": m["content"]}] if m["content"] else []) + \
                         [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                          for c in m["tool_calls"]]
                push("assistant", blocks)
            else:   # tool results travel back to Claude as a "user" block
                push("user", [{"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}])

        tools = [{"name": t.name, "description": t.desc,
                  "input_schema": {"type": "object", "required": list(t.params),
                                   "properties": {k: {"type": "string", "description": v}
                                                  for k, v in t.params.items()}}}
                 for t in agent.tools]

        # >>> SENDING TO CLAUDE <<<
        #   system   = this agent's prompt                         (PER AGENT)
        #   tools    = this agent's tools                          (PER AGENT)
        #   messages = `out` = this agent's WHOLE persistent diary (PER AGENT, across ALL its visits;
        #              other agents' diaries are never included). Grows with every visit.
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
    """Print each agent's saved diary so you can SEE the separation and the persistence."""
    print(f"\n{'─'*70}\nSAVED DIARIES for {tid}\n{'─'*70}")
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

    # Visit #1 to the buyer: a normal purchase.
    chat("A) first purchase (buyer's FIRST visit)", "thread-1", [
        "Research Apple and buy some if it looks good",
        "AAPL",
        "5 shares, market order",         # -> approval prompt
        "yes",                            # -> order runs
    ], store, brain)

    # Visit #2 to the SAME buyer, same thread. "the same" only makes sense if the buyer remembers.
    chat("B) follow-up on the same thread (buyer's SECOND visit)", "thread-1", [
        "Buy 3 more of the same",         # supervisor forwards this verbatim; buyer resolves "the same"
        "yes",
    ], store, brain)

    dump(store, "thread-1")               # <- the buyer's diary now holds BOTH visits
