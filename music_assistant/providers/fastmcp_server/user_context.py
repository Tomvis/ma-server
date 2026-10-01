"""
User-context middleware: run every MCP request as the user its Bearer token belongs to.

``MASTokenVerifier`` only authenticates the token; nothing bound the resulting user to
MA's ``current_user`` context, so every tool ran as anonymous playback. Since music
sources got owners (#6255) anonymous playback may only use sources shared with
everyone, which made ``playback_play_media`` reject anything on a user's own sources.

The user is resolved inside the FastMCP request (not in the verifier) because stateful
streamable-HTTP sessions execute tools in a session task that does not inherit the
HTTP request's context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware

from music_assistant.controllers.webserver.helpers.auth_middleware import current_user

if TYPE_CHECKING:
    from fastmcp.server.middleware.middleware import CallNext, MiddlewareContext

    from music_assistant.mass import MusicAssistant


class CurrentUserMiddleware(Middleware):  # type: ignore[misc, unused-ignore]
    """Bind the token's MA user to ``current_user`` for the duration of each request."""

    def __init__(self, mass: MusicAssistant) -> None:
        """
        Initialise the middleware.

        :param mass: MusicAssistant instance used to look up the token's user.
        """
        self._mass = mass

    async def on_request(
        self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]
    ) -> Any:
        """
        Run the request as the authenticated user; anonymous when there is none.

        :param context: The FastMCP middleware context.
        :param call_next: The next handler in the chain.
        """
        token = get_access_token()
        # MASTokenVerifier puts the MA user id in client_id
        user = await self._mass.webserver.auth.get_user(token.client_id) if token else None
        reset = current_user.set(user)
        try:
            return await call_next(context)
        finally:
            current_user.reset(reset)
