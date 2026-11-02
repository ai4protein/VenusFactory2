"""An accepted turn belongs to the server, not to its SSE subscriber."""
import asyncio
import logging
from starlette.responses import StreamingResponse

_RUNNING = set()
_END = object()
log = logging.getLogger(__name__)


class BackgroundStreamResponse(StreamingResponse):
    def __init__(self, source, **kwargs):
        self.queue = asyncio.Queue(maxsize=128)
        self.detached = asyncio.Event()
        # Keep a strong reference until the producer has finalized session state
        # and released its session lock. Disconnect only stops the subscriber.
        self.producer = asyncio.create_task(self._produce(source))
        _RUNNING.add(self.producer)
        self.producer.add_done_callback(_RUNNING.discard)
        super().__init__(self._subscribe(), **kwargs)

    async def _offer(self, frame):
        if self.detached.is_set():
            return
        put = asyncio.create_task(self.queue.put(frame))
        detached = asyncio.create_task(self.detached.wait())
        try:
            await asyncio.wait((put, detached), return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (put, detached):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put, detached, return_exceptions=True)

    async def _produce(self, source):
        try:
            async for frame in source:
                await self._offer(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Background chat stream failed")
            await self._offer('event: error\ndata: {"message":"Could not finish this turn."}\n\n')
        finally:
            if hasattr(source, "aclose"):
                await source.aclose()
            await self._offer(_END)

    async def _subscribe(self):
        try:
            while True:
                frame = await self.queue.get()
                if frame is _END:
                    return
                yield frame
        finally:
            self.detached.set()

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.detached.set()
