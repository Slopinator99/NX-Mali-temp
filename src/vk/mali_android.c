/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Android only: the HAL module the Android Vulkan loader opens, and the
 * VK_ANDROID_native_buffer entry points it drives swapchains with. The
 * work is in mali_wsi.c; this file unwraps Android's structures.
 *
 * The loader (/system/lib64/libvulkan.so) loads the driver with
 * android_load_sphal_library or android_dlopen_ext (adrenotools hooks both
 * to substitute this library), looks up the symbol HMI, and opens device
 * "vk0" through it; from then on it only uses the three functions of
 * hwvulkan_device_t and vkGetInstanceProcAddr. Surfaces and swapchains
 * are the loader's; per swapchain image it calls vkCreateImage with a
 * VkNativeBufferANDROID, per acquire vkAcquireImageANDROID, per present
 * vkQueueSignalReleaseImageANDROID.
 */

#include <errno.h>
#include <math.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <poll.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>
#include <unistd.h>

#include <hardware/hwvulkan.h>

#include "util/log.h"
#include "util/u_atomic.h"
#include "util/u_call_once.h"
#include "vk_log.h"
#include "vk_util.h"

#include "mali_image.h"
#include "mali_cmd_buffer.h"
#include "mali_memory.h"
#include "mali_vk.h"
#include "mali_wsi.h"

#define wsi_errorf(obj, result, ...)                                            \
   (mesa_loge("malisx2: WSI: " __VA_ARGS__), vk_errorf(obj, result, __VA_ARGS__))

/* ---------------------------------------------------------------------- */
/* The HAL module                                                          */

static int mali_hal_open(const struct hw_module_t *mod, const char *id,
                         struct hw_device_t **dev);

static_assert(HWVULKAN_DISPATCH_MAGIC == ICD_LOADER_MAGIC,
              "dispatchable handles start with the loader magic");

static hw_module_methods_t mali_hal_methods = {
   .open = mali_hal_open,
};

__attribute__((visibility("default"))) struct hwvulkan_module_t HAL_MODULE_INFO_SYM = {
   .common = {
      .tag = HARDWARE_MODULE_TAG,
      .module_api_version = HWVULKAN_MODULE_API_VERSION_0_1,
      .hal_api_version = HARDWARE_MAKE_API_VERSION(1, 0),
      .id = HWVULKAN_HARDWARE_MODULE_ID,
      .name = "malisx2 Vulkan driver (kbase)",
      .author = "ARMSX2 and bmdhacks",
      .methods = &mali_hal_methods,
   },
};

static int
mali_hal_close(struct hw_device_t *dev)
{
   /* Called when the loader unloads the driver. */
   free(container_of(dev, hwvulkan_device_t, common));
   return 0;
}

static int
mali_hal_open(const struct hw_module_t *mod, const char *id, struct hw_device_t **dev)
{
   if (mod != &HAL_MODULE_INFO_SYM.common || strcmp(id, HWVULKAN_DEVICE_0) != 0)
      return -1;

   hwvulkan_device_t *hal = calloc(1, sizeof(*hal));
   if (!hal)
      return -1;
   *hal = (hwvulkan_device_t){
      .common = {
         .tag = HARDWARE_DEVICE_TAG,
         .version = HWVULKAN_DEVICE_API_VERSION_0_1,
         .module = &HAL_MODULE_INFO_SYM.common,
         .close = mali_hal_close,
      },
      .EnumerateInstanceExtensionProperties =
         (PFN_vkEnumerateInstanceExtensionProperties)vk_icdGetInstanceProcAddr(
            NULL, "vkEnumerateInstanceExtensionProperties"),
      .CreateInstance =
         (PFN_vkCreateInstance)vk_icdGetInstanceProcAddr(NULL, "vkCreateInstance"),
      .GetInstanceProcAddr =
         (PFN_vkGetInstanceProcAddr)vk_icdGetInstanceProcAddr(NULL, "vkGetInstanceProcAddr"),
   };
   *dev = &hal->common;
   return 0;
}

/* ---------------------------------------------------------------------- */
/* Gralloc usage                                                           */

static char no_afbc_property[PROP_VALUE_MAX];
static util_once_flag no_afbc_once = UTIL_ONCE_FLAG_INIT;

static void
no_afbc_usage_init(void)
{
   bool fallback;
   __system_property_get("ro.vendor.arm.gralloc.no_afbc_usage_flags", no_afbc_property);
   mali_wsi_no_afbc_usage(no_afbc_property, &fallback);
   if (fallback)
      mesa_logw("malisx2: ro.vendor.arm.gralloc.no_afbc_usage_flags is not readable or "
                "not a usable value (\"%s\")", no_afbc_property);
}

uint64_t
mali_wsi_swapchain_no_afbc(unsigned arch)
{
   util_call_once(&no_afbc_once, no_afbc_usage_init);
   return mali_wsi_arch_no_afbc_usage(arch, no_afbc_property);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetSwapchainGrallocUsage2ANDROID(VkDevice _device, VkFormat format,
                                      VkImageUsageFlags imageUsage,
                                      VkSwapchainImageUsageFlagsANDROID swapchainImageUsage,
                                      uint64_t *grallocConsumerUsage,
                                      uint64_t *grallocProducerUsage)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   /* Shared (front-buffer) presentation is not offered: sharedImage is
    * false, so the loader never asks for it. */
   if (swapchainImageUsage & VK_SWAPCHAIN_IMAGE_USAGE_SHARED_BIT_ANDROID)
      return wsi_errorf(dev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                       "shared presentable images are not supported");
   const VkResult result =
      mali_wsi_gralloc_usage(mali_device_physical(dev), format, imageUsage,
                             mali_wsi_swapchain_no_afbc(mali_device_physical(dev)->arch),
                             grallocConsumerUsage, grallocProducerUsage);
   /* Once per process: what gralloc is asked for decides whether a buffer
    * comes back linear, so it belongs next to the buffers' own log line. */
   static int logged_usage;
   if (result == VK_SUCCESS && p_atomic_cmpxchg(&logged_usage, 0, 1) == 0)
      mesa_logi("malisx2: WSI: gralloc usage for format %d, image usage 0x%x: consumer 0x%llx, "
                "producer 0x%llx (no-AFBC property \"%s\")", format, imageUsage,
                (unsigned long long)*grallocConsumerUsage,
                (unsigned long long)*grallocProducerUsage, no_afbc_property);
   return result;
}

/* The gralloc0 form: 32 bits, so no room for the no-AFBC bit (the blob
 * drops it too). The loader uses it only if ...Usage2 is missing. */
VKAPI_ATTR VkResult VKAPI_CALL
mali_GetSwapchainGrallocUsageANDROID(VkDevice _device, VkFormat format,
                                     VkImageUsageFlags imageUsage, int *grallocUsage)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   uint64_t consumer, producer;
   VkResult result = mali_wsi_gralloc_usage(mali_device_physical(dev), format, imageUsage,
                                            0, &consumer, &producer);
   if (result == VK_SUCCESS)
      *grallocUsage = (int)(consumer | producer);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Swapchain images                                                        */

/*
 * What the loader handed over, for the log: the handle's counts and ints
 * (gralloc keeps format, size and usage in them, and the layout of the
 * buffer, AFBC or not), and the native buffer's own fields. Logged when a
 * bind fails, and once for the first buffer of the process.
 */
static void
log_native_buffer(const VkNativeBufferANDROID *nb, const struct mali_image *image, bool error)
{
   const native_handle_t *h = nb->handle;
   char ints[48 * 9 + 1] = "";
   char line[768];
   const int n = h->numInts < 48 ? h->numInts : 48;

   for (int i = 0; i < n; i++)
      snprintf(ints + strlen(ints), sizeof(ints) - strlen(ints), "%x ", h->data[h->numFds + i]);
   snprintf(line, sizeof(line),
            "malisx2: WSI: native buffer for a %ux%u image: stride %d, gralloc format %d, usage "
            "0x%x, usage3 0x%llx, handle version %d, %d fd(s), %d int(s) [%s%s]",
            image->vk.extent.width, image->vk.extent.height, nb->stride, nb->format,
            (unsigned)nb->usage, (unsigned long long)nb->usage3, h->version, h->numFds,
            h->numInts, ints, h->numInts > n ? "..." : "");
   if (error)
      mesa_loge("%s", line);
   else
      mesa_logi("%s", line);
}

VkResult
mali_android_bind_native_buffer(struct mali_device *dev, struct mali_image *image,
                                const VkNativeBufferANDROID *nb,
                                const VkAllocationCallbacks *alloc)
{
   static int logged_first;

   /* native_handle_t is its three counts and then the fds and ints, all
    * ints; mali_wsi.c reads it as that. */
   static_assert(sizeof(native_handle_t) == 3 * sizeof(int) &&
                    offsetof(native_handle_t, data) == 3 * sizeof(int),
                 "native_handle_t is three ints and then the data");

   if (!nb->handle)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "native buffer without a dma-buf fd");

   /* Which fd is the buffer, and the stride when the gralloc does not
    * report one, are decided there: see mali_wsi.c. */
   const VkResult result =
      mali_wsi_image_bind_native_handle(dev, image, (const int *)nb->handle, nb->stride, alloc);
   if (result != VK_SUCCESS)
      log_native_buffer(nb, image, true);
   else if (p_atomic_cmpxchg(&logged_first, 0, 1) == 0)
      log_native_buffer(nb, image, false);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Acquire and present                                                     */

VKAPI_ATTR VkResult VKAPI_CALL
mali_AcquireImageANDROID(VkDevice _device, VkImage image, int nativeFenceFd,
                         VkSemaphore semaphore, VkFence fence)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   return mali_wsi_acquire(dev, nativeFenceFd, semaphore, fence);
}

/* ---------------------------------------------------------------------- */
/* DIAGNOSTIC: present-time pixel check (flicker hunt)                     */

/*
 * For every vkQueueSignalReleaseImageANDROID the image's dma-buf, geometry
 * and release fence are handed to one worker thread. It waits for the fence
 * (the work covered by the present), maps the buffer read-only and samples a
 * grid of pixels. All-zero RGB over the grid means the GPU finished the frame
 * and it is black: that separates "the GPU rendered black" from "the picture
 * was fine and went black later (buffer hand-off / compositor)". Only black
 * runs and a periodic summary are logged; the present path itself never
 * blocks and holds no lock while this runs.
 */
uint32_t mali_dbg_cur;
uint32_t mali_dbg_ev[64][24];
uint32_t mali_dbg_cb[64][8];
struct mali_dbg_cp mali_dbg_cp[64][64];
uint32_t mali_dbg_ncp[64];
uint32_t mali_dbg_ncb[64];

#define PCHK_QUEUE 16
#define PCHK_GRID 24

struct pchk_job {
   int dmabuf;      /* dup, owned by the job */
   int fence;       /* dup of the release fence, or -1 */
   uint32_t width, height, row_stride;
   uint64_t size;
   uint32_t seq;
};

static struct {
   pthread_mutex_t lock;
   pthread_cond_t cond;
   struct pchk_job q[PCHK_QUEUE];
   uint32_t head, count;
   bool started, disabled;
   /* stats (worker thread only, except dropped/skipped) */
   uint32_t presents, black, black_run, single_black, nofence, timeouts;
   uint32_t dropped, skipped, logged;
} pchk = {
   .lock = PTHREAD_MUTEX_INITIALIZER,
   .cond = PTHREAD_COND_INITIALIZER,
};

static bool
pchk_sample_black(const struct pchk_job *j, uint32_t *lit_out, uint64_t *luma_out)
{
   void *map = mmap(NULL, j->size, PROT_READ, MAP_SHARED, j->dmabuf, 0);
   if (map == MAP_FAILED) {
      if (pchk.logged++ < 3)
         mesa_logw("malisx2: present-check: mmap of the swapchain dma-buf failed: %s",
                   strerror(errno));
      pchk.disabled = true;
      return false;
   }
   struct dma_buf_sync sync = {.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ};
   ioctl(j->dmabuf, DMA_BUF_IOCTL_SYNC, &sync);

   uint32_t lit = 0;
   uint64_t luma = 0;
   for (uint32_t gy = 0; gy < PCHK_GRID; gy++) {
      const uint32_t y = (uint64_t)(2 * gy + 1) * j->height / (2 * PCHK_GRID);
      const uint8_t *row = (const uint8_t *)map + (uint64_t)y * j->row_stride;
      for (uint32_t gx = 0; gx < PCHK_GRID; gx++) {
         const uint32_t x = (uint64_t)(2 * gx + 1) * j->width / (2 * PCHK_GRID);
         const uint8_t *px = row + (uint64_t)x * 4;
         luma += px[0] + px[1] + px[2];
         if (px[0] | px[1] | px[2])
            lit++;
      }
   }

   sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
   ioctl(j->dmabuf, DMA_BUF_IOCTL_SYNC, &sync);
   munmap(map, j->size);
   *lit_out = lit;
   *luma_out = luma;
   return true;
}

static void *
pchk_worker(void *arg)
{
   uint32_t prev_seq = 0;
   uint64_t la[3] = {0, 0, 0};
   uint32_t pulses = 0;
   for (;;) {
      pthread_mutex_lock(&pchk.lock);
      while (!pchk.count)
         pthread_cond_wait(&pchk.cond, &pchk.lock);
      struct pchk_job j = pchk.q[pchk.head];
      pchk.head = (pchk.head + 1) % PCHK_QUEUE;
      pchk.count--;
      pthread_mutex_unlock(&pchk.lock);

      bool fence_ok = true;
      if (j.fence >= 0) {
         struct pollfd pfd = {.fd = j.fence, .events = POLLIN};
         const int r = poll(&pfd, 1, 500);
         if (r <= 0) {
            fence_ok = false;
            pchk.timeouts++;
         }
         close(j.fence);
      } else {
         pchk.nofence++;
      }

      uint32_t lit = 0;
      uint64_t luma = 0;
      if (!pchk.disabled && fence_ok && pchk_sample_black(&j, &lit, &luma)) {
         /* DIAG v32: a frame brighter/darker than both neighbours (which agree). */
         la[0] = la[1]; la[1] = la[2]; la[2] = luma;
         if (la[0] && la[1] && la[2]) {
            const double ref = (double)(la[0] + la[2]) / 2.0;
            const double side = fabs((double)la[0] - (double)la[2]) / ref;
            const double dev = ((double)la[1] - ref) / ref;
            if (side < 0.004 && fabs(dev) > 0.008 && fabs(dev) < 0.5 && pulses++ < 40) {
               mesa_logw("malisx2: pchk-pulse present#%u dev=%+.2f%% (sum %llu vs %llu/%llu) frame seq=%u",
                         pchk.presents + 1, dev * 100.0, (unsigned long long)la[1],
                         (unsigned long long)la[0], (unsigned long long)la[2], prev_seq);
               for (int d = -2; d <= 1; d++) {
                  const uint32_t sq = prev_seq + d;
                  if (pchk.presents > 3000 && (d == 0 || d == -1)) {
                     const uint32_t ncp = mali_dbg_ncp[sq % 64] < 64 ? mali_dbg_ncp[sq % 64] : 64;
                     for (uint32_t q = 0; q < ncp; q += 6) {
                        char line[256];
                        int pos = 0;
                        for (uint32_t z = q; z < q + 6 && z < ncp; z++)
                           pos += snprintf(line + pos, sizeof(line) - pos, " %08x:%x:%08x",
                                           mali_dbg_cp[sq % 64][z].key, mali_dbg_cp[sq % 64][z].size,
                                           mali_dbg_cp[sq % 64][z].hash);
                        mesa_logw("malisx2: pchk-cp seq=%u %s #%u%s", sq, d == 0 ? "PULSE" : "ok", q, line);
                     }
                  }
                  {
                     const uint32_t *e = mali_dbg_ev[sq % 64];
                     mesa_logw("malisx2: pchk-pulse-ev seq=%u %s img+%u/-%u view+%u/-%u mem+%u/-%u buf+%u/-%u "
                               "flush=%u bind=%u cpbuf=%u buf2img=%u cpimg=%u blit=%u clrimg=%u disp=%u "
                               "fill=%u upd=%u barr=%u bindpipe=%u beginrend=%u clratt=%u img2buf=%u resolve=%u",
                               sq, d == 0 ? "PULSE" : "ok", e[0], e[1], e[2], e[3], e[4], e[5], e[6],
                               e[7], e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15], e[16],
                               e[17], e[18], e[19], e[20], e[21], e[22], e[23]);
                  }
                  const uint32_t *c = mali_dbg_cb[sq % 64];
                  mesa_logw("malisx2: pchk-pulse-cb seq=%u %s n=%u: %08x %08x %08x %08x %08x %08x",
                            sq, d == 0 ? "PULSE" : "ok", mali_dbg_ncb[sq % 64], c[0], c[1], c[2],
                            c[3], c[4], c[5]);
               }
            }
         }
         pchk.presents++;
         const bool is_black = lit == 0;
         if (is_black) {
            pchk.black++;
            pchk.black_run++;
         } else if (pchk.black_run) {
            if (pchk.black_run == 1) {
               pchk.single_black++;
               if (pchk.single_black <= 12) {
                  for (int d = -2; d <= 0; d++) {
                     const uint32_t sq = prev_seq + d;
                     {
                        const uint32_t *ev = mali_dbg_ev[sq % 64];
                        mesa_logw("malisx2: pchk-ev seq=%u %s img+%u/-%u view+%u/-%u mem+%u/-%u buf+%u/-%u flush=%u bind=%u",
                                  sq, d == 0 ? "BLACK" : "ok", ev[0], ev[1], ev[2], ev[3], ev[4],
                                  ev[5], ev[6], ev[7], ev[8], ev[9]);
                     }
                  }
               }
            }
            if (pchk.logged++ < 80)
               mesa_logw("malisx2: present-check: BLACK run of %u presented frame(s) ended "
                         "at present #%u (lit=%u/%u, single-frame runs so far %u)",
                         pchk.black_run, pchk.presents, lit, PCHK_GRID * PCHK_GRID,
                         pchk.single_black);
            pchk.black_run = 0;
         }
         if (pchk.presents % 600 == 0)
            mesa_logw("malisx2: present-check: presents=%u black=%u single-frame-runs=%u "
                      "nofence=%u fence-timeouts=%u dropped=%u skipped=%u",
                      pchk.presents, pchk.black, pchk.single_black, pchk.nofence,
                      pchk.timeouts, pchk.dropped, pchk.skipped);
      }
      close(j.dmabuf);
      prev_seq = j.seq;
   }
   return NULL;
}

static void
pchk_submit(VkImage image, int release_fence)
{
   VK_FROM_HANDLE(mali_image, img, image);
   if (!img || !img->wsi_mem || img->wsi_mem->dmabuf_fd < 0 || pchk.disabled)
      return;
   switch (img->vk.format) {
   case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
      break;
   default:
      pchk.skipped++;
      return;
   }
   const uint32_t cur_seq = mali_dbg_cur;
   memset(mali_dbg_ev[(cur_seq + 1) % 64], 0, sizeof(mali_dbg_ev[0]));
   mali_dbg_ncb[(cur_seq + 1) % 64] = 0;
   mali_dbg_ncp[(cur_seq + 1) % 64] = 0;
   __atomic_store_n(&mali_dbg_cur, cur_seq + 1, __ATOMIC_RELAXED);
   struct pchk_job j = {
      .seq = cur_seq,
      .dmabuf = fcntl(img->wsi_mem->dmabuf_fd, F_DUPFD_CLOEXEC, 0),
      .fence = release_fence >= 0 ? fcntl(release_fence, F_DUPFD_CLOEXEC, 0) : -1,
      .width = img->vk.extent.width,
      .height = img->vk.extent.height,
      .row_stride = img->planes[0].layout.slices[0].row_stride,
      .size = img->size,
   };
   if (j.dmabuf < 0 || !j.row_stride || !j.width || !j.height ||
       (uint64_t)j.row_stride * j.height > j.size) {
      if (j.dmabuf >= 0)
         close(j.dmabuf);
      if (j.fence >= 0)
         close(j.fence);
      pchk.skipped++;
      return;
   }

   pthread_mutex_lock(&pchk.lock);
   if (!pchk.started) {
      pthread_t t;
      if (pthread_create(&t, NULL, pchk_worker, NULL) == 0) {
         pthread_detach(t);
         pchk.started = true;
      } else {
         pchk.disabled = true;
      }
   }
   if (pchk.disabled || pchk.count == PCHK_QUEUE) {
      pchk.dropped++;
      pthread_mutex_unlock(&pchk.lock);
      close(j.dmabuf);
      if (j.fence >= 0)
         close(j.fence);
      return;
   }
   pchk.q[(pchk.head + pchk.count) % PCHK_QUEUE] = j;
   pchk.count++;
   pthread_cond_signal(&pchk.cond);
   pthread_mutex_unlock(&pchk.lock);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_QueueSignalReleaseImageANDROID(VkQueue _queue, uint32_t waitSemaphoreCount,
                                    const VkSemaphore *pWaitSemaphores, VkImage image,
                                    int *pNativeFenceFd)
{
   VK_FROM_HANDLE(mali_queue, queue, _queue);
   const VkResult result =
      mali_wsi_release(queue, waitSemaphoreCount, pWaitSemaphores, pNativeFenceFd);
   if (result == VK_SUCCESS)
      pchk_submit(image, pNativeFenceFd ? *pNativeFenceFd : -1);
   return result;
}
