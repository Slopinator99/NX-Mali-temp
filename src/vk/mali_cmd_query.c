/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Occlusion queries on CSF (v11): vkCreateQueryPool, vkDestroyQueryPool,
 * vkGetQueryPoolResults, vkCmdResetQueryPool, vkCmdBeginQuery, vkCmdEndQuery,
 * vkCmdCopyQueryPoolResults. This is the G615 build's query file (the JM
 * timestamps, mali_jm_query.c, are only for the G57/G52 builds): Eden
 * (unlike ARMSX2) creates VK_QUERY_TYPE_OCCLUSION pools, and the G615 is
 * CSF, so the occlusion path must live here.
 *
 * Only OCCLUSION is supported. ARMSX2's other query types (timestamps,
 * VK_QUERY_TYPE_PIPELINE_STATISTICS) are gated on features this driver does
 * not report, so nothing here creates those pools; a pool of an unsupported
 * type returns VK_ERROR_FEATURE_NOT_PRESENT (the crash this file fixes, a
 * waist stub returning the same).
 *
 * Per query, 16 bytes of uncached descriptor-pool memory, in the blob's and
 * panvk's interleaved layout (so a GPU report address and an availability
 * sync-object address are one query's two halves):
 *
 *    query i:  report   at gpu + 16*i      (struct mali_oq_report, 8 B)
 *              avail    at gpu + 16*i + 8  (struct mali_cs_sync32,  8 B)
 *
 * The fragment job's occlusion counter writes the report (sample count) and,
 * once the pass's caches are flushed and the availability sync object is
 * signaled, the uncached availability word is visible to the CPU. Both
 * halves are zeroed on reset (vkCmdResetQueryPool) and on begin (the count
 * always starts at zero).
 *
 * The GPU-side ordering (clear -> fragment accumulates -> flush -> signal
 * availability) is recorded by mali_cmd_draw.c (prepare_draw / prepare_oq)
 * and mali_cmd_render.c (issue_fragment_jobs / cmd_render_end); this file
 * owns the pool, the reset, the two host entry points and the GPU copy.
 */

#include "mali_cmd_state.h"
#include "mali_memory.h"
#include "mali_queue.h"

#include <string.h>

#include "util/os_time.h"
#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_query_pool.h"

struct mali_query_pool {
   struct vk_query_pool vk;
   struct mali_kbase_bo bo; /* vk.query_count * MALI_QUERY_SLOT_SIZE bytes,
                             * CPU+GPU read/write, uncached */
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_query_pool, vk.base, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL)

#define MALI_QUERY_SLOT_SIZE 16u /* report (8) + availability sync32 (8) */

/* The occlusion counter's 64-bit sample count at a query's report address. */
struct mali_oq_report {
   uint64_t value;
};

/* A query's availability object: an 8-byte sync32 (panvk's
 * panvk_query_available_obj, arch >= 10) -- SYNC_SET32 writes seqno, and
 * the CPU reads it (panvk_query_is_available: seqno != 0). */
struct mali_cs_sync32 {
   uint32_t seqno;
   uint32_t error;
};

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateQueryPool)(VkDevice _device, const VkQueryPoolCreateInfo *pCreateInfo,
                               const VkAllocationCallbacks *pAllocator, VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   if (pCreateInfo->queryType != VK_QUERY_TYPE_OCCLUSION)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "only occlusion query pools are supported");

   struct mali_query_pool *pool =
      vk_query_pool_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   const struct mali_kbase_alloc_info ai = {
      .size = (uint64_t)pCreateInfo->queryCount * MALI_QUERY_SLOT_SIZE,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
      .mem_class = MALI_KBASE_MEM_CLASS_DESCRIPTOR_POOL,
   };
   if (mali_kbase_alloc(dev->kbase, &ai, &pool->bo) != MALI_KBASE_SUCCESS) {
      vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   memset(pool->bo.cpu, 0, ai.size);

   *pQueryPool = mali_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(DestroyQueryPool)(VkDevice _device, VkQueryPool _pool,
                                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   if (!pool)
      return;
   mali_kbase_free(dev->kbase, &pool->bo);
   vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
}

/* The query's report address (occlusion counter). */
static inline uint64_t
query_report_addr(const struct mali_query_pool *pool, uint32_t query)
{
   return pool->bo.gpu_va + (uint64_t)query * MALI_QUERY_SLOT_SIZE;
}

/* The query's availability sync-object address (struct mali_cs_sync32). */
static inline uint64_t
query_avail_addr(const struct mali_query_pool *pool, uint32_t query)
{
   return query_report_addr(pool, query) + sizeof(struct mali_oq_report);
}

/* The query's availability word, from the CPU (uncached memory). */
static inline uint64_t
query_avail_value(const struct mali_query_pool *pool, uint32_t query)
{
   const struct mali_cs_sync32 *avail =
      (const struct mali_cs_sync32 *)((uint8_t *)pool->bo.cpu +
                                      (uint64_t)query * MALI_QUERY_SLOT_SIZE +
                                      sizeof(struct mali_oq_report));
   return avail->seqno;
}

/* Zero a contiguous run of reports or availability words on the fragment
 * subqueue (vkCmdResetQueryPool; outside a render pass, Vulkan). The stores
 * are flushed and the caches cleaned so the CPU sees the zeroes before a
 * later query reads them. Mirrors panvk's reset path, unrolled (the run is
 * Eden's 256-query pool; the fixed 8-byte stride makes two stores per query
 * a fixed, small stream).
 */
VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdResetQueryPool)(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                                 uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);

   if (!queryCount)
      return;

   /* Not valid inside a render pass instance (Vulkan); the fragment
    * subqueue is idle here, so its waits are trivial. */
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);

   /* Wait for any prior query operations on this subqueue. */
   cs_wait_slot(b, MALI_SB_DEFERRED_SYNC);

   struct cs_index addr = mali_cs_scratch_reg64(b, 0);
   struct cs_index val = mali_cs_scratch_reg64(b, 2);

   /* Zero every report, then every availability word. The address advances
    * by the slot stride rather than using instruction offsets: STORE's
    * Offset field is a signed 16-bit, which a large pool would overflow. */
   const uint64_t bases[2] = {query_report_addr(pool, firstQuery),
                              query_avail_addr(pool, firstQuery)};
   for (unsigned half = 0; half < 2; half++) {
      cs_move64_to(b, addr, bases[half]);
      cs_move64_to(b, val, 0);
      for (uint32_t i = 0; i < queryCount; i++) {
         cs_store64(b, val, addr, 0);
         cs_add_imm64(b, addr, addr, MALI_QUERY_SLOT_SIZE);
      }
   }
   cs_flush_stores(b);

   /* Make the zeroes visible to the CPU (the host read path spins on the
    * uncached availability word). Wait for any prior fragment work, flush
    * the caches clean, signal IMM_FLUSH, then wait on it so the stores have
    * landed before the stream continues. */
   struct cs_index flush_id = mali_cs_scratch_reg32(b, 4);
   cs_move32_to(b, flush_id, 0);
   cs_flush_caches(b, MALI_CS_FLUSH_MODE_CLEAN, MALI_CS_FLUSH_MODE_CLEAN,
                   MALI_CS_OTHER_FLUSH_MODE_NONE, flush_id,
                   cs_defer(cmd->dev->csf->sb.all_iters_mask, MALI_SB_IMM_FLUSH));
   cs_wait_slot(b, MALI_SB_IMM_FLUSH);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBeginQuery)(VkCommandBuffer commandBuffer, VkQueryPool _pool, uint32_t query,
                             VkQueryControlFlags flags)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   struct mali_gfx_occlusion *oq = &cmd->gfx.occlusion;

   if (!cmd->gfx.spec_render_pass)
      mali_trace_anomaly(cmd, 2, "BeginQuery outside spec pass");
   mali_trace_push(cmd, 'Q');
   oq->ptr = query_report_addr(pool, query);
   oq->syncobj = query_avail_addr(pool, query);
   oq->mode = (flags & VK_QUERY_CONTROL_PRECISE_BIT) ? MALI_OCCLUSION_MODE_COUNTER
                                                     : MALI_OCCLUSION_MODE_PREDICATE;
   /* The next draw re-emits the OQ report pointer and the counting DCD0. */
   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_OQ;

   /* From the Vulkan spec: "When an occlusion query begins, the count of
    * passing samples always starts at zero." Clear the report on the
    * fragment subqueue; the accumulation happens in the pass's fragment job. */
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);
   struct cs_index addr = mali_cs_scratch_reg64(b, 0);
   struct cs_index zero = mali_cs_scratch_reg64(b, 2);
   cs_move64_to(b, addr, oq->ptr);
   cs_move64_to(b, zero, 0);
   cs_store64(b, zero, addr, 0);
   cs_flush_stores(b);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdEndQuery)(VkCommandBuffer commandBuffer, VkQueryPool _pool, uint32_t query)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   struct mali_gfx_occlusion *oq = &cmd->gfx.occlusion;
   struct mali_render_state *r = &cmd->gfx.render;
   const uint64_t syncobj = query_avail_addr(pool, query);

   if (!cmd->gfx.spec_render_pass)
      mali_trace_anomaly(cmd, 3, "EndQuery outside spec pass");
   mali_trace_push(cmd, 'q');

   /* Stop the accumulation for the next draw. */
   oq->ptr = 0;
   oq->syncobj = 0;
   oq->mode = MALI_OCCLUSION_MODE_DISABLED;
   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_OQ;

   /* If this is the query the current pass's fragment job writes, the pass
    * end wraps it into the pending list and signals it after that job
    * (panvk). Otherwise its report is already written: a prior pass's job,
    * or no draw used it since begin (its count is the zero begin stored).
    * Flush the caches, then signal availability after the flush. */
   if (r->oq.last == syncobj)
      return;

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);
   struct cs_index flush_id = mali_cs_scratch_reg32(b, 0);
   struct cs_index val = mali_cs_scratch_reg32(b, 1);
   struct cs_index addr = mali_cs_scratch_reg64(b, 2);

   cs_move32_to(b, flush_id, 0);
   cs_flush_caches(b, MALI_CS_FLUSH_MODE_CLEAN, MALI_CS_FLUSH_MODE_CLEAN,
                   MALI_CS_OTHER_FLUSH_MODE_NONE, flush_id,
                   cs_defer(cmd->dev->csf->sb.all_iters_mask, MALI_SB_DEFERRED_FLUSH));

   cs_move32_to(b, val, 1);
   cs_move64_to(b, addr, syncobj);
   cs_sync32_set(b, true, MALI_CS_SYNC_SCOPE_CSG, val, addr,
                 cs_defer(MALI_SB_MASK(MALI_SB_DEFERRED_FLUSH), MALI_SB_DEFERRED_SYNC));
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(GetQueryPoolResults)(VkDevice _device, VkQueryPool _pool, uint32_t firstQuery,
                                   uint32_t queryCount, size_t dataSize, void *pData,
                                   VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   VkResult result = VK_SUCCESS;
   const int64_t start = os_time_get_nano();

   for (uint32_t i = 0; i < queryCount; i++) {
      const uint32_t q = firstQuery + i;
      uint8_t *out = (uint8_t *)pData + i * stride;
      uint64_t avail;

      for (;;) {
         avail = query_avail_value(pool, q);
         if (avail || !(flags & VK_QUERY_RESULT_WAIT_BIT))
            break;
         /* CSF has no mali_jm_wait_locked; poll the uncached availability
          * word with a timeout so a lost job cannot hang the host. */
         if (dev->lost)
            return vk_error(dev, VK_ERROR_DEVICE_LOST);
         if (os_time_get_nano() - start > 2 * 1000 * 1000 * 1000ll)
            return vk_errorf(dev, VK_ERROR_DEVICE_LOST,
                             "occlusion query wait timed out");
      }

      if (avail || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
         const uint64_t value =
            *(const uint64_t *)((uint8_t *)pool->bo.cpu + (uint64_t)q * MALI_QUERY_SLOT_SIZE);
         if (flags & VK_QUERY_RESULT_64_BIT)
            *(uint64_t *)out = value;
         else
            *(uint32_t *)out = (uint32_t)value;
      } else {
         result = VK_NOT_READY;
      }

      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
         void *out_avail = out + (flags & VK_QUERY_RESULT_64_BIT ? 8 : 4);
         if (flags & VK_QUERY_RESULT_64_BIT)
            *(uint64_t *)out_avail = avail ? 1 : 0;
         else
            *(uint32_t *)out_avail = avail ? 1 : 0;
      }
   }
   return result;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdCopyQueryPoolResults)(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                                       uint32_t firstQuery, uint32_t queryCount,
                                       VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                       VkDeviceSize dstStride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   VK_FROM_HANDLE(mali_buffer, dbuf, dstBuffer);

   if (!queryCount)
      return;

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);

   /* VK_QUERY_RESULT_WAIT_BIT: wait for the queries to become available
    * before reading them. */
   if (flags & VK_QUERY_RESULT_WAIT_BIT)
      cs_wait_slot(b, MALI_SB_DEFERRED_SYNC);

   struct cs_index src = mali_cs_scratch_reg64(b, 0);
   struct cs_index dst = mali_cs_scratch_reg64(b, 2);
   struct cs_index res = mali_cs_scratch_reg64(b, 4);
   struct cs_index avail = mali_cs_scratch_reg32(b, 6);

   cs_move64_to(b, src, query_report_addr(pool, firstQuery));
   cs_move64_to(b, dst, mali_buffer_gpu_va(dbuf, dstOffset));
   const bool is64 = (flags & VK_QUERY_RESULT_64_BIT) != 0;
   const bool with_avail = (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0;

   /* Both addresses advance per query: STORE/LOAD's Offset field is a
    * signed 16-bit, which a large pool or an app-chosen stride would
    * overflow. */
   for (uint32_t i = 0; i < queryCount; i++) {
      if (is64)
         cs_load64_to(b, res, src, 0);
      else
         cs_load32_to(b, res, src, 0);
      if (with_avail)
         cs_load32_to(b, avail, src, sizeof(struct mali_oq_report));

      if (is64)
         cs_store64(b, res, dst, 0);
      else
         cs_store32(b, res, dst, 0);
      /* Availability follows the result: 8 bytes after a 64-bit result, 4
       * after a 32-bit one (VkQueryResultFlags layout). */
      if (with_avail)
         cs_store32(b, avail, dst, is64 ? 8 : 4);

      cs_add_imm64(b, src, src, MALI_QUERY_SLOT_SIZE);
      cs_add_imm64(b, dst, dst, dstStride);
   }
   cs_flush_stores(b);
}
