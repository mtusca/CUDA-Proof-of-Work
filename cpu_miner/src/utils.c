#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>

// Compute SHA256 and convert to hex
void apply_sha256(const BYTE *input, BYTE *output) {
    size_t input_length = strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}

// Constructs a Merkle tree and returns the root hash
void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]) {
    BYTE (*hashes)[SHA256_HASH_SIZE] = (BYTE (*)[SHA256_HASH_SIZE])malloc(max_transactions_in_a_block * SHA256_HASH_SIZE);
    if (!hashes) {
        fprintf(stderr, "Error: Unable to allocate memory for hashes\n");
        exit(EXIT_FAILURE);
    }

    // Compute the SHA256 hash for each transaction
    for (int i = 0; i < n; i++) {
        apply_sha256(transactions + i * transaction_size, hashes[i]);
    }

    // Build the Merkle tree
    while (n > 1) {
        int new_n = 0;
        for (int i = 0; i < n; i += 2) {
            BYTE combined[SHA256_HASH_SIZE * 2];
            if (i + 1 < n) {
                strcpy((char *)combined, (const char *)hashes[i]);
                strcat((char *)combined, (const char *)hashes[i+1]);
            } else {
                // If odd number of hashes, duplicate the last one
                strcpy((char *)combined, (const char *)hashes[i]);
                strcat((char *)combined, (const char *)hashes[i]);
            }
            apply_sha256(combined, hashes[new_n++]);
        }
        n = new_n;
    }

    memcpy(merkle_root, hashes[0], SHA256_HASH_SIZE);

    free(hashes);
}

int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    char nonce_string[NONCE_SIZE];

    for (uint32_t nonce = 0; nonce <= max_nonce; nonce++) {
        sprintf(nonce_string, "%u", nonce);
        strcpy((char *)block_content + current_length, nonce_string);
        apply_sha256(block_content, block_hash);

        if (compare_hashes(block_hash, difficulty) <= 0) {
            *valid_nonce = nonce;
            return 0;
        }
    }

    return 1;
}

void warm_up_gpu() {}
