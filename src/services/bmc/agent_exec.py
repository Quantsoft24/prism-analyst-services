"""Run an ADK agent to completion and return its final text.

BMC's per-block sub-agents + the reconciler are deterministic one-shot agent
runs (not interactive chat), so they don't need the SSE ``AgentRunner``. This
helper stands up an ADK ``Runner`` with a fresh in-memory session, runs the
agent, and returns the concatenated final text.

We DRAIN the event stream (never ``break`` early) — same lesson as the chat
path: breaking mid-stream triggers OpenTelemetry context-detach noise. ADK
stops yielding right after the final response anyway.
"""

from __future__ import annotations

import uuid

from src.agents.base import PrismAgent


async def run_agent_to_text(agent: PrismAgent, user_message: str) -> str:
    """Run ``agent`` on ``user_message`` to completion; return final text.

    Each call uses an isolated session (BMC block generation is stateless per
    block). Tool calls the agent makes (e.g. NRE ``compute_*``) happen inside
    the ADK loop transparently.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    adk_agent = agent.build()
    runner = Runner(
        agent=adk_agent,
        app_name="prism-bmc",
        session_service=InMemorySessionService(),
    )

    user_id = "bmc"
    session_id = f"bmc_{uuid.uuid4().hex[:12]}"
    await runner.session_service.create_session(
        app_name="prism-bmc", user_id=user_id, session_id=session_id
    )

    new_message = genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])

    final_parts: list[str] = []
    final_seen = False
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=new_message
    ):
        if final_seen:
            continue  # drain remaining events for clean span teardown
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                final_parts.append(text)
        check = getattr(event, "is_final_response", None)
        if callable(check):
            try:
                if check():
                    final_seen = True
            except Exception:
                pass

    return "".join(final_parts).strip()
