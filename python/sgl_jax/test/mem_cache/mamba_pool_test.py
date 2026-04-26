"""Unit tests for RecurrentStatePool and HybridReqToTokenPool."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool, RecurrentStatePool


class TestRecurrentStatePool(unittest.TestCase):
    def setUp(self):
        self.pool = RecurrentStatePool(
            size=4,
            num_layers=2,
            conv_dim=6,
            conv_state_len=3,
            num_heads=2,
            head_k_dim=4,
            head_v_dim=4,
            dtype=jnp.bfloat16,
        )

    def test_alloc_and_free_sequence(self):
        # Slot 0 is the null slot, so allocs should return indices in [1, 4].
        slots = self.pool.alloc(3)
        self.assertEqual(slots, [1, 2, 3])
        self.assertEqual(self.pool.available_size(), 1)
        # Free middle slot; next alloc must reuse it (FIFO from tail of extend).
        self.pool.free([2])
        self.assertEqual(self.pool.available_size(), 2)
        next_two = self.pool.alloc(2)
        self.assertEqual(set(next_two), {2, 4})

    def test_alloc_too_many_returns_none(self):
        self.assertIsNone(self.pool.alloc(99))

    def test_get_write_roundtrip(self):
        slots = self.pool.alloc(2)
        idx = jnp.asarray(slots, dtype=jnp.int32)
        conv_in, rec_in = self.pool.get_layer(0, idx)
        self.assertEqual(conv_in.shape, (2, 6, 3))
        self.assertEqual(rec_in.shape, (2, 2, 4, 4))
        # Zero-initialised on alloc.
        np.testing.assert_array_equal(np.asarray(conv_in), 0)
        np.testing.assert_array_equal(np.asarray(rec_in), 0)

        new_conv = jnp.ones((2, 6, 3), dtype=jnp.bfloat16)
        new_rec = jnp.ones((2, 2, 4, 4), dtype=jnp.float32) * 2.5
        self.pool.write_layer(0, idx, new_conv, new_rec)
        conv_in2, rec_in2 = self.pool.get_layer(0, idx)
        np.testing.assert_allclose(np.asarray(conv_in2), 1.0, rtol=0, atol=1e-3)
        np.testing.assert_allclose(np.asarray(rec_in2), 2.5, rtol=0, atol=1e-6)
        # Other layer must remain zero.
        conv_l1, rec_l1 = self.pool.get_layer(1, idx)
        np.testing.assert_array_equal(np.asarray(conv_l1), 0)
        np.testing.assert_array_equal(np.asarray(rec_l1), 0)

    def test_free_zeros_slots(self):
        slots = self.pool.alloc(2)
        idx = jnp.asarray(slots, dtype=jnp.int32)
        self.pool.write_layer(
            0,
            idx,
            jnp.ones((2, 6, 3), dtype=jnp.bfloat16),
            jnp.ones((2, 2, 4, 4), dtype=jnp.float32),
        )
        self.pool.free(slots)
        # Fresh fetch of the just-freed slots must show zeros (free must clear
        # the stored state; free_slots order is not part of the contract).
        conv_after, rec_after = self.pool.get_layer(0, idx)
        np.testing.assert_array_equal(np.asarray(conv_after), 0)
        np.testing.assert_array_equal(np.asarray(rec_after), 0)


class TestHybridReqToTokenPool(unittest.TestCase):
    def _make(self):
        return HybridReqToTokenPool(
            size=4,
            max_context_len=32,
            num_linear_layers=2,
            conv_dim=6,
            conv_state_len=3,
            num_heads=2,
            head_k_dim=4,
            head_v_dim=4,
            state_dtype=jnp.bfloat16,
        )

    def test_alloc_returns_parallel_slots(self):
        pool = self._make()
        req_slots = pool.alloc(3)
        self.assertEqual(len(req_slots), 3)
        self.assertTrue(set(req_slots).issubset({0, 1, 2, 3}))
        mamba_slots = pool.get_mamba_indices(req_slots)
        # Every mamba slot must be a valid non-null slot.
        self.assertTrue(np.all(mamba_slots >= 1))
        self.assertTrue(np.all(mamba_slots <= 4))

    def test_free_rolls_back_both_pools(self):
        pool = self._make()
        req_slots = pool.alloc(4)
        self.assertIsNotNone(req_slots)
        self.assertEqual(pool.recurrent_state_pool.available_size(), 0)
        pool.free(req_slots)
        self.assertEqual(pool.recurrent_state_pool.available_size(), 4)
        self.assertEqual(pool.available_size(), 4)
        # req_to_mamba must be zeroed for freed reqs.
        self.assertTrue(np.all(pool.req_to_mamba == 0))

    def test_recurrent_state_pool_exhaustion_rolls_back_req(self):
        pool = self._make()
        # Manually drain mamba pool.
        pool.recurrent_state_pool.alloc(4)
        # Now super().alloc can still succeed, but recurrent_state_pool.alloc must fail.
        result = pool.alloc(1)
        self.assertIsNone(result)
        # Req pool slots must have been rolled back.
        self.assertEqual(pool.available_size(), 4)


if __name__ == "__main__":
    unittest.main()
