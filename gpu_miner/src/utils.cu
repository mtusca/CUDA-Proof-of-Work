#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include "utils.h"
#include "sha256.h"

static constexpr int TPB   = 256;
static constexpr int MAX_B = 65535;

#define HASH_ASCII  SHA256_HASH_SIZE

__device__ __constant__ BYTE d_diff[32];

__device__ __forceinline__
void bytes_to_hex(const BYTE *in, char *out)
{
    const char lut[] = "0123456789abcdef";
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        BYTE b = in[i];
        out[i*2]     = lut[b >> 4];
        out[i*2 + 1] = lut[b & 0x0f];
    }
    out[64] = '\0';
}

__device__ __forceinline__
void sha256_hex(const BYTE *msg, int len, char *ascii_out)
{
    BYTE dig[32];
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, msg, len);
    sha256_final(&ctx, dig);
    bytes_to_hex(dig, ascii_out);
}

__device__ __forceinline__
int u32_to_dec(uint32_t v, char *out)
{
    int len = 0;
    do { out[len++] = '0' + (v % 10); v /= 10; } while (v);
    for (int i = 0; i < len/2; ++i) {
        char t = out[i];
        out[i] = out[len-1-i];
        out[len-1-i] = t;
    }
    return len;
}

__global__ void k_leaf_hash(const char *tx, int txsz, char *leaf, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    sha256_hex(reinterpret_cast<const BYTE*>(tx + id*txsz),
               txsz-1, leaf + id*HASH_ASCII);
}

__global__ void k_merkle_level(const char *prev, char *next,
                               int pairs, int odd)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= pairs) return;

    const char *left  = prev + id*2*HASH_ASCII;
    const char *right = (id == pairs-1 && odd)
                       ? left
                       : left + HASH_ASCII;

    char concat[128];
    memcpy(concat    , left , 64);
    memcpy(concat+64 , right, 64);

    sha256_hex(reinterpret_cast<const BYTE*>(concat), 128,
               next + id*HASH_ASCII);
}

__global__ void k_nonce(const char *header, int hlen, uint32_t max_n,
                        uint32_t *g_nonce, BYTE *g_hash, int *g_found)
{
    uint64_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t step = (uint64_t)gridDim.x * blockDim.x;

    char buf[160];
    BYTE dig[32];
    memcpy(buf, header, hlen);

    for (uint64_t n64 = tid; n64 <= max_n && !(*g_found); n64 += step) {
        uint32_t n = (uint32_t)n64;
        int len = u32_to_dec(n, buf + hlen);
        buf[hlen + len] = '\0';

        SHA256_CTX ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, reinterpret_cast<BYTE*>(buf), hlen + len);
        sha256_final(&ctx, dig);

        bool ok = true;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            BYTE a = dig[i], b = d_diff[i];
            if (a < b) break;
            if (a > b) { ok = false; break; }
        }
        if (ok && atomicCAS(g_found, 0, 1) == 0) {
            *g_nonce = n;
            memcpy(g_hash, dig, 32);
            return;
        }
    }
}

void construct_merkle_root(int txsz, BYTE *tx, int max_tx_block, int n,
                           BYTE root_ascii[HASH_ASCII])
{
    char *d_tx;         cudaMalloc(&d_tx, n * txsz);
    cudaMemcpy(d_tx, tx, n * txsz, cudaMemcpyHostToDevice);

    char *d_a, *d_b;
    cudaMalloc(&d_a, max_tx_block * HASH_ASCII);
    cudaMalloc(&d_b, max_tx_block * HASH_ASCII);

    int blocks = (n + TPB - 1) / TPB;
    k_leaf_hash<<<blocks, TPB>>>(d_tx, txsz, d_a, n);
    cudaDeviceSynchronize();

    int cur = n;
    char *curp = d_a, *nextp = d_b;
    while (cur > 1) {
        int pairs = (cur + 1) >> 1;
        blocks = (pairs + TPB - 1) / TPB;
        k_merkle_level<<<blocks, TPB>>>(curp, nextp, pairs, cur & 1);
        cudaDeviceSynchronize();
        cur = pairs;
        char *tmp = curp;
        curp = nextp;
        nextp = tmp;
    }

    cudaMemcpy(root_ascii, curp, HASH_ASCII, cudaMemcpyDeviceToHost);
    cudaFree(d_tx);
    cudaFree(d_a);
    cudaFree(d_b);
}

int find_nonce(BYTE *diff_hex, uint32_t max_n,
               BYTE *header, size_t hlen,
               BYTE *hash_hex, uint32_t *nonce_out)
{
    BYTE diff_bin[32];
    for (int i = 0; i < 32; ++i) {
        unsigned x;
        sscanf(reinterpret_cast<char*>(diff_hex) + 2*i, "%2x", &x);
        diff_bin[i] = (BYTE)x;
    }
    cudaMemcpyToSymbol(d_diff, diff_bin, 32);

    char *d_hdr;       cudaMalloc(&d_hdr, hlen + 1);
    cudaMemcpy(d_hdr, header, hlen, cudaMemcpyHostToDevice);

    uint32_t *d_nonce;
    BYTE *d_hash;
    int *d_found;
    cudaMalloc(&d_nonce, sizeof(uint32_t));
    cudaMalloc(&d_hash , 32);
    cudaMallocManaged(&d_found, sizeof(int));
    *d_found = 0;

    uint64_t total   = (uint64_t)max_n + 1;
    uint64_t blocks64 = (total + TPB - 1) / TPB;
    int blocks = blocks64 > MAX_B ? MAX_B : (int)blocks64;

    k_nonce<<<blocks, TPB>>>(d_hdr, (int)hlen, max_n,
                             d_nonce, d_hash, d_found);
    cudaDeviceSynchronize();

    int rc = 1;
    if (*d_found) {
        BYTE dig[32];
        cudaMemcpy(nonce_out, d_nonce, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(dig      , d_hash , 32, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 32; ++i) {
            sprintf(reinterpret_cast<char*>(hash_hex) + i*2, "%02x", dig[i]);
        }
        hash_hex[64] = '\0';
        rc = 0;
    }

    cudaFree(d_hdr);
    cudaFree(d_nonce);
    cudaFree(d_hash);
    cudaFree(d_found);
    return rc;
}

void warm_up_gpu()
{
    cudaFree(0);
}
