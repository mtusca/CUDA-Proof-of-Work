#ifndef UTILS_H
#define UTILS_H

#ifdef DEBUG
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(...)
#endif

#include "sha256.h"
#include <stdint.h>

#define SHA256_HASH_SIZE 65 // 32 byte array is 64 characters long + 1 for null terminator
#define NONCE_SIZE 11 // UINT32_MAX is 10 chars long + 1 for null terminator
#define BLOCK_SIZE (2 * (SHA256_HASH_SIZE - 1) + (NONCE_SIZE - 1) + 1) // Prev. block hash + Merke root Top Hash + Nonce + 1 for null terminator

void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]);
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce);
void warm_up_gpu();

#endif // UTILS_H
