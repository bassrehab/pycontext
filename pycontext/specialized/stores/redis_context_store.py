"""
pycontext/specialized/stores/redis_context_store.py

Redis-based distributed context store for PyContext.
"""
from typing import Dict, List, Optional, Any, Union
import json
import asyncio
import pickle
from datetime import datetime

from ...core.mcp.protocol import (
    ContextType,
    ContextBlock,
    ContextMetrics,
    ContextPackage
)


class RedisContextStore:
    """
    Distributed context store using Redis.
    Allows context packages to be shared across multiple processes or machines.
    """

    def __init__(self, redis_client, ttl: int = 3600, prefix: str = "pycontext:"):
        """
        Initialize the Redis context store.

        Args:
            redis_client: Redis client (aioredis or redis)
            ttl: Default time-to-live in seconds
            prefix: Key prefix for Redis keys
        """
        self.redis = redis_client
        self.ttl = ttl
        self.prefix = prefix
        self.is_async = self._check_if_async()

    def _check_if_async(self) -> bool:
        """Check if the Redis client is async."""
        # This is a simple heuristic - might need adjustment for different Redis clients
        client_module = self.redis.__class__.__module__
        return "aio" in client_module.lower()

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"{self.prefix}session:{session_id}"

    def _block_key(self, session_id: str, block_id: str) -> str:
        """Generate Redis key for a block."""
        return f"{self.prefix}block:{session_id}:{block_id}"

    def _blocks_key(self, session_id: str) -> str:
        """Generate Redis key for a session's block list."""
        return f"{self.prefix}blocks:{session_id}"

    def _serialize_block(self, block: ContextBlock) -> bytes:
        """Serialize a context block."""
        return pickle.dumps(block)

    def _deserialize_block(self, data: bytes) -> ContextBlock:
        """Deserialize a context block."""
        return pickle.loads(data)

    async def store_session_async(self, session: ContextPackage) -> None:
        """
        Store session metadata in Redis (async version).

        Args:
            session: Context package to store
        """
        session_key = self._session_key(session.session_id)
        blocks_key = self._blocks_key(session.session_id)

        # Store main session metadata
        session_data = {
            "agent_id": session.agent_id,
            "version": str(session.version),
            "trace_id": session.trace_id,
            "timestamp": str(datetime.now().timestamp()),
            "max_tokens": str(session.max_tokens),
            "metrics": json.dumps({
                "total_tokens": session.metrics.total_tokens,
                "context_saturation": session.metrics.context_saturation,
                "type_distribution": session.metrics.type_distribution
            })
        }

        # Create pipeline for better performance
        async with self.redis.pipeline() as pipe:
            # Store session metadata
            await pipe.hmset(session_key, session_data)
            await pipe.expire(session_key, self.ttl)

            # Store block IDs
            await pipe.delete(blocks_key)
            if session.blocks:
                await pipe.sadd(blocks_key, *[block.id for block in session.blocks])
                await pipe.expire(blocks_key, self.ttl)

            # Store each block separately
            for block in session.blocks:
                block_key = self._block_key(session.session_id, block.id)
                await pipe.set(block_key, self._serialize_block(block))
                await pipe.expire(block_key, self.ttl)

            # Execute pipeline
            await pipe.execute()

    def store_session_sync(self, session: ContextPackage) -> None:
        """
        Store session metadata in Redis (sync version).

        Args:
            session: Context package to store
        """
        session_key = self._session_key(session.session_id)
        blocks_key = self._blocks_key(session.session_id)

        # Store main session metadata
        session_data = {
            "agent_id": session.agent_id,
            "version": str(session.version),
            "trace_id": session.trace_id,
            "timestamp": str(datetime.now().timestamp()),
            "max_tokens": str(session.max_tokens),
            "metrics": json.dumps({
                "total_tokens": session.metrics.total_tokens,
                "context_saturation": session.metrics.context_saturation,
                "type_distribution": session.metrics.type_distribution
            })
        }

        # Create pipeline for better performance
        pipe = self.redis.pipeline()

        # Store session metadata
        pipe.hmset(session_key, session_data)
        pipe.expire(session_key, self.ttl)

        # Store block IDs
        pipe.delete(blocks_key)
        if session.blocks:
            pipe.sadd(blocks_key, *[block.id for block in session.blocks])
            pipe.expire(blocks_key, self.ttl)

        # Store each block separately
        for block in session.blocks:
            block_key = self._block_key(session.session_id, block.id)
            pipe.set(block_key, self._serialize_block(block))
            pipe.expire(block_key, self.ttl)

        # Execute pipeline
        pipe.execute()

    async def retrieve_session_async(self, session_id: str) -> Optional[ContextPackage]:
        """
        Retrieve complete session from Redis (async version).

        Args:
            session_id: Session identifier

        Returns:
            Context package if found, None otherwise
        """
        session_key = self._session_key(session_id)
        blocks_key = self._blocks_key(session_id)

        # Check if session exists
        exists = await self.redis.exists(session_key)
        if not exists:
            return None

        # Get session metadata
        session_data = await self.redis.hgetall(session_key)

        if not session_data:
            return None

        # Convert bytes to strings
        session_data = {k.decode() if isinstance(k, bytes) else k:
                            v.decode() if isinstance(v, bytes) else v
                        for k, v in session_data.items()}

        # Get block IDs
        block_ids = await self.redis.smembers(blocks_key)
        block_ids = [bid.decode() if isinstance(bid, bytes) else bid for bid in block_ids]

        # Get blocks
        blocks = []
        for block_id in block_ids:
            block_key = self._block_key(session_id, block_id)
            block_data = await self.redis.get(block_key)
            if block_data:
                block = self._deserialize_block(block_data)
                blocks.append(block)

        # Parse metrics
        metrics_json = session_data.get("metrics", "{}")
        metrics_dict = json.loads(metrics_json)

        metrics = ContextMetrics(
            total_tokens=metrics_dict.get("total_tokens", 0),
            context_saturation=metrics_dict.get("context_saturation", 0.0),
            type_distribution=metrics_dict.get("type_distribution", {})
        )

        # Create context package
        package = ContextPackage(
            session_id=session_id,
            agent_id=session_data.get("agent_id", "unknown"),
            blocks=blocks,
            metrics=metrics,
            version=int(session_data.get("version", "1")),
            trace_id=session_data.get("trace_id"),
            max_tokens=int(session_data.get("max_tokens", "8192"))
        )

        return package

    def retrieve_session_sync(self, session_id: str) -> Optional[ContextPackage]:
        """
        Retrieve complete session from Redis (sync version).

        Args:
            session_id: Session identifier

        Returns:
            Context package if found, None otherwise
        """
        session_key = self._session_key(session_id)
        blocks_key = self._blocks_key(session_id)

        # Check if session exists
        exists = self.redis.exists(session_key)
        if not exists:
            return None

        # Get session metadata
        session_data = self.redis.hgetall(session_key)

        if not session_data:
            return None

        # Convert bytes to strings
        session_data = {k.decode() if isinstance(k, bytes) else k:
                            v.decode() if isinstance(v, bytes) else v
                        for k, v in session_data.items()}

        # Get block IDs
        block_ids = self.redis.smembers(blocks_key)
        block_ids = [bid.decode() if isinstance(bid, bytes) else bid for bid in block_ids]

        # Get blocks
        blocks = []
        for block_id in block_ids:
            block_key = self._block_key(session_id, block_id)
            block_data = self.redis.get(block_key)
            if block_data:
                block = self._deserialize_block(block_data)
                blocks.append(block)

        # Parse metrics
        metrics_json = session_data.get("metrics", "{}")
        metrics_dict = json.loads(metrics_json)

        metrics = ContextMetrics(
            total_tokens=metrics_dict.get("total_tokens", 0),
            context_saturation=metrics_dict.get("context_saturation", 0.0),
            type_distribution=metrics_dict.get("type_distribution", {})
        )

        # Create context package
        package = ContextPackage(
            session_id=session_id,
            agent_id=session_data.get("agent_id", "unknown"),
            blocks=blocks,
            metrics=metrics,
            version=int(session_data.get("version", "1")),
            trace_id=session_data.get("trace_id"),
            max_tokens=int(session_data.get("max_tokens", "8192"))
        )

        return package

    async def update_block_async(
            self,
            session_id: str,
            block: ContextBlock
    ) -> bool:
        """
        Update a specific block in a session (async version).

        Args:
            session_id: Session identifier
            block: Updated block

        Returns:
            Whether the update was successful
        """
        session_key = self._session_key(session_id)
        block_key = self._block_key(session_id, block.id)
        blocks_key = self._blocks_key(session_id)

        # Check if session exists
        exists = await self.redis.exists(session_key)
        if not exists:
            return False

        # Update block
        await self.redis.set(block_key, self._serialize_block(block))
        await self.redis.expire(block_key, self.ttl)

        # Ensure block ID is in the blocks set
        await self.redis.sadd(blocks_key, block.id)
        await self.redis.expire(blocks_key, self.ttl)

        return True

    def update_block_sync(
            self,
            session_id: str,
            block: ContextBlock
    ) -> bool:
        """
        Update a specific block in a session (sync version).

        Args:
            session_id: Session identifier
            block: Updated block

        Returns:
            Whether the update was successful
        """
        session_key = self._session_key(session_id)
        block_key = self._block_key(session_id, block.id)
        blocks_key = self._blocks_key(session_id)

        # Check if session exists
        exists = self.redis.exists(session_key)
        if not exists:
            return False

        # Update block
        self.redis.set(block_key, self._serialize_block(block))
        self.redis.expire(block_key, self.ttl)

        # Ensure block ID is in the blocks set
        self.redis.sadd(blocks_key, block.id)
        self.redis.expire(blocks_key, self.ttl)

        return True

    async def delete_session_async(self, session_id: str) -> bool:
        """
        Delete a session and all its blocks (async version).

        Args:
            session_id: Session identifier

        Returns:
            Whether the deletion was successful
        """
        session_key = self._session_key(session_id)
        blocks_key = self._blocks_key(session_id)

        # Check if session exists
        exists = await self.redis.exists(session_key)
        if not exists:
            return False

        # Get block IDs
        block_ids = await self.redis.smembers(blocks_key)
        block_ids = [bid.decode() if isinstance(bid, bytes) else bid for bid in block_ids]

        # Create pipeline for better performance
        async with self.redis.pipeline() as pipe:
            # Delete blocks
            for block_id in block_ids:
                block_key = self._block_key(session_id, block_id)
                await pipe.delete(block_key)

            # Delete blocks set and session
            await pipe.delete(blocks_key)
            await pipe.delete(session_key)

            # Execute pipeline
            await pipe.execute()

        return True

    def delete_session_sync(self, session_id: str) -> bool:
        """
        Delete a session and all its blocks (sync version).

        Args:
            session_id: Session identifier

        Returns:
            Whether the deletion was successful
        """
        session_key = self._session_key(session_id)
        blocks_key = self._blocks_key(session_id)

        # Check if session exists
        exists = self.redis.exists(session_key)
        if not exists:
            return False

        # Get block IDs
        block_ids = self.redis.smembers(blocks_key)
        block_ids = [bid.decode() if isinstance(bid, bytes) else bid for bid in block_ids]

        # Create pipeline for better performance
        pipe = self.redis.pipeline()

        # Delete blocks
        for block_id in block_ids:
            block_key = self._block_key(session_id, block_id)
            pipe.delete(block_key)

        # Delete blocks set and session
        pipe.delete(blocks_key)
        pipe.delete(session_key)

        # Execute pipeline
        pipe.execute()

        return True

    async def list_sessions_async(self) -> List[str]:
        """
        List all session IDs (async version).

        Returns:
            List of session IDs
        """
        pattern = f"{self.prefix}session:*"
        keys = await self.redis.keys(pattern)

        # Extract session IDs from keys
        sessions = []
        prefix_len = len(f"{self.prefix}session:")

        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            session_id = key_str[prefix_len:]
            sessions.append(session_id)

        return sessions

    def list_sessions_sync(self) -> List[str]:
        """
        List all session IDs (sync version).

        Returns:
            List of session IDs
        """
        pattern = f"{self.prefix}session:*"
        keys = self.redis.keys(pattern)

        # Extract session IDs from keys
        sessions = []
        prefix_len = len(f"{self.prefix}session:")

        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            session_id = key_str[prefix_len:]
            sessions.append(session_id)

        return sessions

    # Unified methods that detect whether to use sync or async
    def store_session(self, session: ContextPackage) -> Union[None, asyncio.Future]:
        """
        Store a session, automatically choosing sync or async based on Redis client.

        Args:
            session: Context package to store

        Returns:
            None for sync, Future for async
        """
        if self.is_async:
            return self.store_session_async(session)
        else:
            return self.store_session_sync(session)

    def retrieve_session(self, session_id: str) -> Union[Optional[ContextPackage], asyncio.Future]:
        """
        Retrieve a session, automatically choosing sync or async based on Redis client.

        Args:
            session_id: Session identifier

        Returns:
            Context package (or None) for sync, Future for async
        """
        if self.is_async:
            return self.retrieve_session_async(session_id)
        else:
            return self.retrieve_session_sync(session_id)

    def update_block(self, session_id: str, block: ContextBlock) -> Union[bool, asyncio.Future]:
        """
        Update a block, automatically choosing sync or async based on Redis client.

        Args:
            session_id: Session identifier
            block: Updated block

        Returns:
            Boolean for sync, Future for async
        """
        if self.is_async:
            return self.update_block_async(session_id, block)
        else:
            return self.update_block_sync(session_id, block)

    def delete_session(self, session_id: str) -> Union[bool, asyncio.Future]:
        """
        Delete a session, automatically choosing sync or async based on Redis client.

        Args:
            session_id: Session identifier

        Returns:
            Boolean for sync, Future for async
        """
        if self.is_async:
            return self.delete_session_async(session_id)
        else:
            return self.delete_session_sync(session_id)

    def list_sessions(self) -> Union[List[str], asyncio.Future]:
        """
        List sessions, automatically choosing sync or async based on Redis client.

        Returns:
            List of session IDs for sync, Future for async
        """
        if self.is_async:
            return self.list_sessions_async()
        else:
            return self.list_sessions_sync()
