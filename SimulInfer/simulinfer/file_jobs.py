"""Short HTTP polling for file translation; event/audio payloads spool to disk."""
import asyncio
import json
import secrets
import tempfile
import time


class Job:
    def __init__(self):
        self.events = tempfile.TemporaryFile(mode='w+b')
        self.offsets = []
        self.done = False
        self.touched = time.monotonic()
        self.finished = None
        self.task = None

    async def emit(self, event):
        self.events.seek(0, 2)
        self.offsets.append(self.events.tell())
        self.events.write(json.dumps(event, ensure_ascii=False).encode() + b'\n')

    def poll(self, cursor, max_bytes=512*1024):
        if type(cursor) is not int or not 0 <= cursor <= len(self.offsets):
            raise ValueError('无效的任务游标')
        self.touched = time.monotonic()
        rows, size, end = [], 0, cursor
        if cursor < len(self.offsets):
            self.events.seek(self.offsets[cursor])
            while end < len(self.offsets) and len(rows) < 128:
                line = self.events.readline()
                if rows and size + len(line) > max_bytes:
                    break
                rows.append(json.loads(line));size += len(line);end += 1
        return {'events': rows, 'next_cursor': end, 'total_events': len(self.offsets), 'done': self.done}


class FileJobs:
    def __init__(self, *, max_jobs=64, max_pending_bytes=256*1024*1024,
                 idle_seconds=120, retention_seconds=900):
        self.jobs = {}
        self.max_jobs = max_jobs
        self.max_pending_bytes = max_pending_bytes
        self.pending_bytes = 0
        self.idle_seconds = idle_seconds
        self.retention_seconds = retention_seconds
        self.cleaner = None

    def create(self, size, run, timeout):
        if len(self.jobs) >= self.max_jobs or self.pending_bytes + size > self.max_pending_bytes:
            raise ValueError('文件任务队列已满，请稍后重试')
        token, job = secrets.token_urlsafe(24), Job()
        self.jobs[token] = job
        self.pending_bytes += size

        async def work():
            try:
                await asyncio.wait_for(run(job.emit), timeout)
            except asyncio.CancelledError:
                await job.emit({'event': 'error', 'detail': '文件任务已取消或长时间未读取'})
                raise
            except Exception as exc:
                await job.emit({'event': 'error', 'detail': str(exc) or type(exc).__name__})
            finally:
                job.done = True
                job.finished = time.monotonic()

        def settled(task):
            # Also release accounting when cancellation preceded the coroutine's first step.
            self.pending_bytes -= size
            if not job.done:
                job.done = True;job.finished = time.monotonic()

        job.task = asyncio.create_task(work())
        job.task.add_done_callback(settled)
        if self.cleaner is None:
            self.cleaner = asyncio.create_task(self.cleanup_loop())
        return token

    def get(self, token):
        job = self.jobs.get(token)
        if job is None:raise KeyError('任务不存在、已过期或服务已重启')
        return job

    async def cancel(self, token):
        job = self.get(token)
        job.task.cancel()
        await asyncio.gather(job.task, return_exceptions=True)

    async def cleanup_loop(self):
        while True:
            await asyncio.sleep(15)
            now = time.monotonic()
            for token, job in list(self.jobs.items()):
                if not job.done and now - job.touched > self.idle_seconds:
                    await self.cancel(token)
                if job.done and now - job.finished > self.retention_seconds:
                    self.jobs.pop(token, None);job.events.close()

    async def close(self):
        if self.cleaner:
            self.cleaner.cancel()
            await asyncio.gather(self.cleaner, return_exceptions=True)
        for job in self.jobs.values():job.task.cancel()
        await asyncio.gather(*(job.task for job in self.jobs.values()), return_exceptions=True)
        for job in self.jobs.values():job.events.close()
        self.jobs.clear()
