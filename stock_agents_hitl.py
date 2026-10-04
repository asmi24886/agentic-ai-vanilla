"""
MULTI-AGENT STOCK DEMO WITH HUMAN-IN-THE-LOOP  (pure Python, no framework)

FEATURES
  1. Supervisor hands off to sub-agents (researcher, buyer); they hand back to the supervisor.
  2. Any sub-agent can PAUSE to ask the user a clarifying question, then RESUME
     when the user answers (ask_user).
  3. The buyer's RISKY tool (place_order) can never run without the user's approval.
     The user can reply: "yes" (approve), "no" (reject), or free text like
     "make it 5 shares" (request a change). That is a PAUSE/RESUME too.
  4. Both pauses are human-in-the-loop (HITL): the graph stops, the human decides, the graph continues.
  5. The server is STATELESS: everything lives in a store, loaded/saved on every request.

TWO KINDS OF PAUSE (both stored in state["pending"])
  * "question"  -> the agent chose to ask. The user's answer becomes the ask_user tool's result.
  * "approval"  -> the ENGINE intercepted a risky tool call BEFORE running it. The user's
                   reply decides whether it runs. The agent cannot skip or bypass this,
                   because the check is in our Python code, not in the prompt.
  Analogy: a bank teller (the agent) can prepare a transfer, but the till physically won't open
  until the manager (the human) signs off. The teller can't talk their way past it.

WHAT WE SEND TO CLAUDE ON EACH CALL  (you asked me to flag this everywhere it happens)
  Claude's API has NO memory, so on every single call we re-send the history.
  In this design:
     system prompt  -> PER AGENT   (only the active agent's job description)
     tools          -> PER AGENT   (only the active agent's tools)
     messages       -> ONE SHARED FULL HISTORY (user + supervisor + researcher + buyer + tool results,
                       all of it, every time). It is NOT filtered per agent.
  Consequence: the buyer can "see" what the researcher found, because it is in the shared history.
  Cost: the array grows each turn, so tokens grow. Common alternatives (not implemented here):
     (a) PER-AGENT history: each sub-agent only sees its own messages; cheaper, but you must pass
         context in explicitly (the handoff 'task' text / summary becomes the ONLY context).
     (b) SUMMARY-ONLY: sub-agent works with a private history, only its final summary goes to the supervisor.
  Every spot in the code where messages go to Claude is marked with:  >>> SENDING TO CLAUDE <<<
"""
import json, os, re, uuid
from dataclasses import dataclass, field
from typing import Callable


# ══════════════════════════════════════════════════════════════════════════
# 1. TOOLS -- what agents can actually DO (here: prints stand in for real API calls)
# ══════════════════════════════════════════════════════════════════════════
def get_stock_price(ticker):
    print(f"   [API] fetching price for {ticker}")
    return f"{ticker} trades at $190.12"            # this string is handed back to the LLM

def get_news(ticker):
    print(f"   [API] fetching news for {ticker}")
    return "Headlines: strong iPhone demand, analyst upgrade to Buy"

def place_order(ticker, qty, order_type):
    print(f"   [API] ORDER PLACED -> BUY {qty} {ticker} ({order_type})")   # <- the dangerous side effect
    return f"Order ORD-1001 filled: BUY {qty} {ticker} ({order_type})"


@dataclass
class Tool:
    name: str
    desc: str                 # the LLM reads this to decide when to use the tool
    params: dict              # {argument name: description}
    fn: Callable = None       # None => "control tool", handled by the engine itself
    risky: bool = False       # True => engine pauses for human approval BEFORE running it


# Control tools (no fn): the engine recognises them by name.
ASK  = Tool("ask_user", "Ask the user a clarifying question and wait for the answer.",
            {"question": "The question to ask"})
BACK = Tool("handoff_back", "Finished or blocked: return control to the supervisor.",
            {"summary": "What was done / found, for the supervisor"})


# ══════════════════════════════════════════════════════════════════════════
# 2. AGENTS -- just a name + instructions + allowed tools. No hidden memory.
#    "Switching agent" = same history, different system prompt + different tools.
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
          "Never research or trade yourself. When all work is done, reply to the user."
          + ONE_TOOL,
          [Tool("transfer_to_researcher", "Hand off to the researcher.", {"task": "What to research"}),
           Tool("transfer_to_buyer",      "Hand off to the buyer.",      {"task": "What to buy"})]),

    Agent("researcher",
          "You research stocks. If the ticker is unclear, ask_user. "
          "When done, handoff_back with a summary." + ONE_TOOL,
          [Tool("get_stock_price", "Current price.", {"ticker": "Ticker"}, get_stock_price),
           Tool("get_news",        "Latest news.",   {"ticker": "Ticker"}, get_news),
           ASK, BACK]),

    # NOTE the buyer's prompt no longer says "ask for a yes before trading".
    # We don't rely on the prompt for safety any more; the ENGINE enforces approval (risky=True).
    Agent("buyer",
          "You place buy orders. Collect ticker, quantity and market/limit via ask_user, then call "
          "place_order. The system will ask the human to approve it. If the result says REJECTED, "
          "handoff_back. If it says NOT EXECUTED with a requested change, call place_order again "
          "with the revised details." + ONE_TOOL,
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
        "messages": [],          # the ONE shared history (this is what we send to Claude)
    }

class Store:
    def __init__(self):
        self.db = {}             # thread_id -> JSON string

    def load(self, tid):
        return json.loads(self.db[tid]) if tid in self.db else new_state()

    def save(self, tid, state):
        # Optimistic lock: refuse if someone saved this thread since we loaded it.
        current = json.loads(self.db[tid])["version"] if tid in self.db else 0
        if current != state["version"]:
            raise RuntimeError("concurrent update, retry")
        state["version"] += 1
        self.db[tid] = json.dumps(state)


# ══════════════════════════════════════════════════════════════════════════
# 4. THE ENGINE -- run_turn(): one call == one HTTP request
# ══════════════════════════════════════════════════════════════════════════
def tool_msg(call_id, name, content, agent):
    """A 'tool result' message. Must reuse the call's id so the LLM API accepts the history."""
    return {"role": "tool", "id": call_id, "name": name, "content": content, "agent": agent}


def run_tool(agent, name, args):
    """Run a normal tool, looking it up ONLY in this agent's own tool list."""
    tool = {t.name: t for t in agent.tools}.get(name)
    return tool.fn(**args) if tool else f"Unknown tool {name}"


def classify(text):
    """
    Turn the human's reply to an approval prompt into one of: approve / reject / feedback.
    (In a real UI this would be buttons [Approve] [Reject] [Edit], giving a structured answer
    instead of guessing from text.)
    """
    t = text.strip().lower()
    if t in {"yes", "y", "approve", "ok", "confirm", "go"}: return "approve"
    if t in {"no", "n", "reject", "cancel", "stop"}:        return "reject"
    return "feedback"          # anything else = "I want something different"


def run_turn(tid, user_text, store, brain, max_steps=20):
    # ── STEP 1: LOAD THE TICKET (server remembers nothing between requests) ──
    state = store.load(tid)

    # ── STEP 2: FILE THE USER'S MESSAGE ──────────────────────────────────────
    pending = state["pending"]
    if pending:
        state["pending"] = None                  # we're resuming, so clear the pause marker
        call = pending["call"]                   # the tool call that was paused

        if pending["kind"] == "question":
            # RESUME after the agent asked a question: the user's text IS the ask_user result.
            state["messages"].append(tool_msg(call["id"], "ask_user", user_text, state["active"]))

        else:  # pending["kind"] == "approval"
            # RESUME after a risky tool call was intercepted. The tool has NOT run yet.
            # The user's reply decides what happens; the agent then reads the outcome as the tool result.
            agent = AGENTS[state["active"]]
            decision = classify(user_text)

            if decision == "approve":
                # Run exactly the args that were shown to the human (the stored call),
                # NOT anything the LLM re-generates. What they approved is what runs.
                result = run_tool(agent, call["name"], call["args"])
            elif decision == "reject":
                result = "REJECTED: the user cancelled this action. Do not retry it."
            else:
                result = (f"NOT EXECUTED: the user wants a change instead: \"{user_text}\". "
                          f"Revise and try again, or ask them.")

            state["messages"].append(tool_msg(call["id"], call["name"], result, agent.name))
    else:
        # Normal case: a fresh message from the user.
        state["messages"].append({"role": "user", "content": user_text})

    # ── STEP 3: THE LOOP. Each pass = ask the active agent's LLM what's next, then do it. ──
    # The loop ends (returns to the user) when: the agent speaks plain text, asks a question,
    # or a risky tool needs approval. Handoffs do NOT end it; they just continue the loop.
    for _ in range(max_steps):

        # 3a. Who's in charge right now? Fetch THEIR prompt and THEIR tools.
        agent = AGENTS[state["active"]]

        # 3b. >>> SENDING TO CLAUDE <<<  (inside brain())
        #     messages = state["messages"]  -> the ENTIRE shared history, every agent's turns included.
        #     system   = THIS agent's prompt only;  tools = THIS agent's tools only.
        #     Claude answers with plain text OR a request to call one tool. It runs nothing itself.
        reply = brain(agent, state["messages"])
        calls = reply.get("tool_calls", [])[:1]       # only handle the first tool call

        # 3c. Record the model's reply in the shared history.
        state["messages"].append({"role": "assistant", "agent": agent.name,
                                  "content": reply.get("text", ""), "tool_calls": calls})

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

        # 3f. Handoff supervisor -> sub-agent: just flip `active`.
        if name.startswith("transfer_to_"):
            target = name.removeprefix("transfer_to_")
            state["active"] = target
            result = f"Transferred to {target}. Task: {args['task']}"
            print(f"   -- handoff: {agent.name} -> {target}")

        # 3g. Handoff sub-agent -> supervisor: flip `active` back; the summary is the tool result.
        elif name == "handoff_back":
            state["active"] = "supervisor"
            result = args["summary"]
            print(f"   -- handoff: {agent.name} -> supervisor")

        # 3h. A normal tool (price lookup, news, place_order...).
        else:
            tool = {t.name: t for t in agent.tools}.get(name)

            # PAUSE TYPE 2 -- risky tool: DO NOT RUN IT. Park the call, ask the human.
            # This check lives in code, so even a misbehaving LLM cannot trade without a yes.
            if tool and tool.risky:
                state["pending"] = {"kind": "approval", "call": call}
                store.save(tid, state)
                return (f"⚠️  [{agent.name}] wants to run {name}({args}).\n"
                        f"    Reply 'yes' to approve, 'no' to cancel, "
                        f"or type a change (e.g. 'make it 5 shares').")

            result = run_tool(agent, name, args)       # safe tool: just run it

        # 3i. Answer the tool call in the history, then loop so the (maybe new) active agent reacts.
        state["messages"].append(tool_msg(call["id"], name, result, agent.name))

    store.save(tid, state)
    return "Stopped: too many steps."


# ══════════════════════════════════════════════════════════════════════════
# 5. BRAINS -- the only code that touches an LLM.
#    A brain is any function: (agent, messages) -> {"text": ..., "tool_calls": [...]}
# ══════════════════════════════════════════════════════════════════════════
def call(name, **args):
    """Build a 'please call this tool' reply, the way an LLM would."""
    return {"tool_calls": [{"id": uuid.uuid4().hex[:8], "name": name, "args": args}]}

def result_of(msgs, tool):    # latest RESULT of a tool
    return next(m["content"] for m in reversed(msgs) if m["role"] == "tool" and m["name"] == tool)
def args_of(msgs, tool):      # latest ARGUMENTS a tool was called with
    return next(c["args"] for m in reversed(msgs) for c in m.get("tool_calls", []) if c["name"] == tool)


def scripted_brain(agent, msgs):
    """
    FAKE LLM so the demo runs offline. These if-statements are the choices a real model
    would make. It reads the LAST message to decide the next move.
    (A real model would be handed `msgs` = the full shared history; see ClaudeBrain below.)
    """
    last = msgs[-1]

    if last["role"] == "user":                       # new user message -> supervisor routes it
        return call("transfer_to_researcher", task=last["content"])

    fn = last["name"]                                # last message is a tool result from tool `fn`

    # ---- SUPERVISOR ----
    if agent.name == "supervisor":
        if last["agent"] == "researcher":            # research done -> go buy
            return call("transfer_to_buyer", task="Buy it if research looks good. " + last["content"])
        return {"text": f"Summary: {last['content']}"}      # buyer done -> tell the user

    # ---- RESEARCHER ----
    if agent.name == "researcher":
        if fn == "transfer_to_researcher":           # arrived, ticker unknown -> ask
            return call("ask_user", question="Which ticker symbol do you mean?")
        if fn == "ask_user":                         # got "AAPL" -> price
            return call("get_stock_price", ticker=last["content"].strip().upper())
        if fn == "get_stock_price":
            return call("get_news", ticker=args_of(msgs, "get_stock_price")["ticker"])
        if fn == "get_news":
            return call("handoff_back",
                        summary=f"{result_of(msgs,'get_stock_price')}. {last['content']}. Looks bullish.")

    # ---- BUYER ----
    if agent.name == "buyer":
        if fn == "transfer_to_buyer":                # arrived -> collect details
            return call("ask_user", question="How many shares, and market or limit order?")

        if fn == "ask_user":                         # got details -> go straight to place_order.
            tkr = args_of(msgs, "get_stock_price")["ticker"]     # (engine will intercept it!)
            qty = re.search(r"\d+", last["content"]).group()
            otype = "limit" if "limit" in last["content"].lower() else "market"
            return call("place_order", ticker=tkr, qty=qty, order_type=otype)

        if fn == "place_order":                      # the engine handed us the outcome:
            c = last["content"]
            if c.startswith("NOT EXECUTED"):         #   user asked for a change -> try again
                prev = args_of(msgs, "place_order")
                qty = re.search(r"\d+", c).group()   #   (a real LLM would just understand the text)
                return call("place_order", ticker=prev["ticker"], qty=qty, order_type=prev["order_type"])
            if c.startswith("REJECTED"):             #   user said no -> report back
                return call("handoff_back", summary="User declined. No order placed.")
            return call("handoff_back", summary=c)   #   approved & filled -> report back


class ClaudeBrain:
    """
    The REAL LLM version (sketch, not run in this demo). Its job is converting between
    our simple message format and the Anthropic API format.
    """
    def __init__(self, model="claude-sonnet-5-5"):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def __call__(self, agent, msgs):
        # Convert OUR full shared history into Anthropic's format.
        # `msgs` here is state["messages"]: EVERY agent's messages, not filtered.
        # (To switch to per-agent history you would filter `msgs` right here, e.g. by m["agent"].)
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

        # Describe ONLY this agent's tools. This is what limits what Claude can ask for.
        tools = [{"name": t.name, "description": t.desc,
                  "input_schema": {"type": "object", "required": list(t.params),
                                   "properties": {k: {"type": "string", "description": v}
                                                  for k, v in t.params.items()}}}
                 for t in agent.tools]

        # >>> SENDING TO CLAUDE <<<
        #   system   = this agent's prompt          (PER AGENT)
        #   tools    = this agent's tools           (PER AGENT)
        #   messages = `out` = full shared history  (ALL agents, everything so far, re-sent every call)
        r = self.client.messages.create(model=self.model, max_tokens=1024,
                                        system=agent.prompt, messages=out, tools=tools)

        # Convert Claude's answer back into our simple format.
        return {"text": "".join(b.text for b in r.content if b.type == "text"),
                "tool_calls": [{"id": b.id, "name": b.name, "args": b.input}
                               for b in r.content if b.type == "tool_use"]}


# ══════════════════════════════════════════════════════════════════════════
# 6. DEMO -- each user message is a separate "HTTP request" to the same thread
# ══════════════════════════════════════════════════════════════════════════
def chat(title, tid, user_messages, store, brain):
    print(f"\n{'═'*70}\n{title}\n{'═'*70}")
    for text in user_messages:
        print(f"\nUSER: {text}")
        print("BOT : ", run_turn(tid, text, store, brain))   # rebuilds everything from the store

if __name__ == "__main__":
    brain = ClaudeBrain() if os.getenv("ANTHROPIC_API_KEY") else scripted_brain
    store = Store()                                          # the ONLY thing shared between requests

    # Scenario A: user CHANGES the order, then approves.
    chat("A) change the order, then approve", "thread-A", [
        "Research Apple and buy some if it looks good",
        "AAPL",
        "10 shares, market order",        # -> approval prompt for 10 shares
        "make it 5 shares instead",       # -> feedback: agent revises, new approval prompt
        "yes",                            # -> NOW place_order actually runs
    ], store, brain)

    # Scenario B: user REJECTS. Nothing is bought.
    chat("B) reject the order", "thread-B", [
        "Research Apple and buy some if it looks good",
        "AAPL",
        "10 shares, market order",        # -> approval prompt
        "no",                             # -> rejected; place_order never runs
    ], store, brain)
