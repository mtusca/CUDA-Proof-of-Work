#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <cstring>
#include <cstdio>

#include "utils.h"
#include "sha256.h"

static constexpr int TPB = 256;

#define HASH_ASCII SHA256_HASH_SIZE

__device__ __constant__ BYTE       d_diff[32];
__device__ __constant__ SHA256_CTX d_pref;

__host__ __device__ __forceinline__ void bytes_to_hex(const BYTE* in,
                                                      char*       out) {
    static constexpr char lut[] = "0123456789abcdef";

#pragma unroll
    for (int i = 0; i < 32; ++i) {
        BYTE b       = in[i];
        out[i * 2]   = lut[b >> 4];
        out[i * 2 + 1] = lut[b & 0x0F];
    }
    out[64] = '\0';
}

__device__ __forceinline__ int u32_to_dec(uint32_t v, char* dst) {
    int len = 0;
    do {
        dst[len++] = static_cast<char>('0' + (v % 10));
        v /= 10;
    } while (v);

    for (int i = 0; i < len / 2; ++i) {
        char tmp         = dst[i];
        dst[i]           = dst[len - 1 - i];
        dst[len - 1 - i] = tmp;
    }
    return len;
}

__global__ void k_leaf_hash(const char* tx,
                            int         txsz,
                            char*       leaf,
                            int         n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    const char* msg = tx + id * txsz;
    int         len = 0;

    while (len < txsz && msg[len] != '\0') ++len;

    BYTE        dig[32];
    SHA256_CTX  ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, reinterpret_cast<const BYTE*>(msg), len);
    sha256_final(&ctx, dig);

    bytes_to_hex(dig, leaf + id * HASH_ASCII);
}

__global__ void k_merkle_level(const char* prev,
                               char*       next,
                               int         pairs,
                               int         odd) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= pairs) return;

    const char* L = prev + id * 2 * HASH_ASCII;
    const char* R = (id == pairs - 1 && odd) ? L : L + HASH_ASCII;

    char buf[128];
    memcpy(buf,     L, 64);
    memcpy(buf + 64, R, 64);

    BYTE        dig[32];
    SHA256_CTX  ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, reinterpret_cast<BYTE*>(buf), 128);
    sha256_final(&ctx, dig);

    bytes_to_hex(dig, next + id * HASH_ASCII);
}

__global__ void k_nonce(uint32_t  max_n,
                        uint32_t* g_nonce,
                        BYTE*     g_hash,
                        int*      g_found) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > max_n || *g_found) return;

    char     digits[11];
    uint32_t n   = static_cast<uint32_t>(tid);
    int      len = u32_to_dec(n, digits);

    SHA256_CTX ctx = d_pref;
    sha256_update(&ctx, reinterpret_cast<BYTE*>(digits), len);

    BYTE dig[32];
    sha256_final(&ctx, dig);

    bool ok = true;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        BYTE a = dig[i];
        BYTE b = d_diff[i];
        if (a < b) break;
        if (a > b) {
            ok = false;
            break;
        }
    }

    if (ok && atomicCAS(g_found, 0, 1) == 0) {
        *g_nonce = n;
        memcpy(g_hash, dig, 32);
    }
}

void construct_merkle_root(int  txsz,
                           BYTE* tx,
                           int   max_tx,
                           int   n,
                           BYTE  root_ascii[HASH_ASCII]) {
    char* d_tx;
    cudaMalloc(&d_tx, n * txsz);
    cudaMemcpy(d_tx, tx, n * txsz, cudaMemcpyHostToDevice);

    char* d_a;
    char* d_b;
    cudaMalloc(&d_a, max_tx * HASH_ASCII);
    cudaMalloc(&d_b, max_tx * HASH_ASCII);

    int blocks = (n + TPB - 1) / TPB;
    k_leaf_hash<<<blocks, TPB>>>(d_tx, txsz, d_a, n);
    cudaDeviceSynchronize();

    int   cur   = n;
    char* curp  = d_a;
    char* nextp = d_b;

    while (cur > 1) {
        int pairs = (cur + 1) >> 1;
        blocks    = (pairs + TPB - 1) / TPB;

        k_merkle_level<<<blocks, TPB>>>(curp, nextp, pairs, cur & 1);
        cudaDeviceSynchronize();

        cur   = pairs;
        char* tmp = curp;
        curp  = nextp;
        nextp = tmp;
    }

    cudaMemcpy(root_ascii, curp, HASH_ASCII, cudaMemcpyDeviceToHost);

    cudaFree(d_tx);
    cudaFree(d_a);
    cudaFree(d_b);
}

int find_nonce(BYTE*     diff_hex,
               uint32_t  max_n,
               BYTE*     header,
               size_t    hlen,
               BYTE*     hash_hex,
               uint32_t* nonce_out) {
    BYTE diff_bin[32];

    for (int i = 0; i < 32; ++i) {
        unsigned x;
        sscanf(reinterpret_cast<char*>(diff_hex) + 2 * i, "%2x", &x);
        diff_bin[i] = static_cast<BYTE>(x);
    }
    cudaMemcpyToSymbol(d_diff, diff_bin, 32);

    SHA256_CTX pref;
    sha256_init(&pref);
    sha256_update(&pref, header, hlen);
    cudaMemcpyToSymbol(d_pref, &pref, sizeof(pref));

    uint32_t* d_nonce;
    BYTE*     d_hash;
    int*      d_found;

    cudaMalloc(&d_nonce, 4);
    cudaMalloc(&d_hash, 32);
    cudaMallocManaged(&d_found, 4);
    *d_found = 0;

    cudaDeviceProp prop;
    int            dev;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    int blocks = 8192;
    if (blocks > prop.maxGridSize[0]) blocks = prop.maxGridSize[0];

    k_nonce<<<blocks, TPB>>>(max_n, d_nonce, d_hash, d_found);
    cudaDeviceSynchronize();

    int rc = 1;
    if (*d_found) {
        BYTE dig[32];
        cudaMemcpy(nonce_out, d_nonce, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(dig,       d_hash,  32, cudaMemcpyDeviceToHost);
        bytes_to_hex(dig, reinterpret_cast<char*>(hash_hex));
        rc = 0;
    }

    cudaFree(d_nonce);
    cudaFree(d_hash);
    cudaFree(d_found);
    return rc;
}

void warm_up_gpu() { cudaFree(0); }
