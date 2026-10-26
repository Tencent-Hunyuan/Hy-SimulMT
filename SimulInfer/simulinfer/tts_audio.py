"""Streaming, pitch-preserving tempo adjustment for raw PCM speech."""
import asyncio
import contextlib


class TTSStreamError(RuntimeError):
    pass


class TempoStream:
    def __init__(self, rate, speed, emit):
        self.rate, self.speed, self.emit = rate, speed, emit
        self.process = self.reader = None

    async def start(self):
        import imageio_ffmpeg
        self.process = await asyncio.create_subprocess_exec(
            imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
            '-f', 's16le', '-ar', str(self.rate), '-ac', '1',
            '-probesize', '32', '-analyzeduration', '0', '-i', 'pipe:0',
            '-af', f'atempo={self.speed}', '-f', 's16le', '-ar', str(self.rate),
            '-ac', '1', '-flush_packets', '1', 'pipe:1',
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        tail = b''
        while raw := await self.process.stdout.read(4096):
            tail += raw
            n = len(tail) // 2 * 2
            if n:
                await self.emit(tail[:n])
            tail = tail[n:]
        if tail:
            raise TTSStreamError('Tempo conversion returned incomplete PCM')

    async def write(self, raw):
        if self.reader.done():
            await self.reader
            raise TTSStreamError('Tempo conversion stopped before the audio ended')
        try:
            self.process.stdin.write(raw)
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise TTSStreamError('Tempo conversion stopped unexpectedly') from exc

    async def finish(self):
        self.process.stdin.close()
        await asyncio.wait_for(self.reader, 10)
        status = await asyncio.wait_for(self.process.wait(), 10)
        if status:
            detail = (await self.process.stderr.read()).decode(errors='replace')[-500:]
            raise TTSStreamError('Tempo conversion failed: ' + detail)

    async def close(self):
        if self.process and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.kill()
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.process:
            await self.process.wait()
