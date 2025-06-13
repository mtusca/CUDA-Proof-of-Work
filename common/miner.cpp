#include <stdio.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

double get_wall_time_sec() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

// Read and process the input card
int read_input(const char *input_file_name, BYTE prev_block_hash[SHA256_HASH_SIZE], BYTE difficulty[SHA256_HASH_SIZE],
    uint32_t *max_nonce, int *max_transactions_in_a_block, int *transaction_size, int *number_of_transactions, FILE **input_file) {
    char input_file_path[256];
    snprintf(input_file_path, sizeof(input_file_path), "../tests/%s.in", input_file_name);

    *input_file = fopen(input_file_path, "r");
    if (!*input_file) {
        fprintf(stderr, "Error opening test file: %s\n", input_file_path);
        perror("fopen");
        return -1;
    }

    char first_line[512];
    if (!fgets(first_line, sizeof(first_line), *input_file)) {
        fprintf(stderr, "Error reading first line of test file: %s\n", input_file_path);
        perror("fgets");
        fclose(*input_file);
        return -1;
    }

    int difficulty_zeros;
    if (sscanf(first_line, "%d,%64s,%d,%u,%d,%d",
               number_of_transactions, prev_block_hash, &difficulty_zeros,
               max_nonce, max_transactions_in_a_block, transaction_size) != 6) {
        fprintf(stderr, "Invalid format in test file: %s\n", input_file_path);
        perror("sscanf");
        fclose(*input_file);
        return -1;
    }

    // If max_nonce is 0, set it to UINT32_MAX
    *max_nonce = (*max_nonce == 0) ? UINT32_MAX : *max_nonce;

    // Generate difficulty
    memset(difficulty, 'f', SHA256_HASH_SIZE - 1);
    difficulty[SHA256_HASH_SIZE - 1] = '\0';
    for (int i = 0; i < difficulty_zeros; i++) {
        difficulty[i] = '0';
    }

    return 0;
}

// Reads transactions from a file and stores them in the transactions array
void read_transactions(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int *txs_read, FILE *file) {
    char buffer[transaction_size + 1];

    while (*txs_read < max_transactions_in_a_block && fgets(buffer, sizeof(buffer), file)) {
        buffer[strcspn(buffer, "\n")] = '\0';
        if (strlen(buffer) > 0) {
            strncpy((char *)(transactions + (*txs_read) * transaction_size), buffer, transaction_size - 1);
            (*txs_read)++;
        };
    }
}

void print_mining_results(const char *input_file_name, int block_number, uint32_t nonce, BYTE *block_hash, double merkle_root_time, double find_nonce_time, double total_block_time, int nonce_found) {
    char output_file_path[256];
    snprintf(output_file_path, sizeof(output_file_path), "output/%s.out", input_file_name);

    FILE *output_file = fopen(output_file_path, "a");
    if (!output_file) {
        fprintf(stderr, "Error opening output file: %s\n", output_file_path);
        perror("fopen");
        return;
    }

    if (nonce_found) {
        fprintf(output_file, "%d,%u,%s,%.5f,%.5f,%.5f\n", block_number + 1, nonce, block_hash, merkle_root_time, find_nonce_time, total_block_time);
    } else {
        fprintf(output_file, "%d,No valid nonce found\n", block_number + 1);
    }

    fclose(output_file);
}

void print_total_time(const char *input_file_name, double total_time, double total_merkle_root_time, double total_find_nonce_time) {
    char output_file_path[256];
    snprintf(output_file_path, sizeof(output_file_path), "output/%s.out", input_file_name);

    FILE *output_file = fopen(output_file_path, "a");
    if (!output_file) {
        fprintf(stderr, "Error opening output file: %s\n", output_file_path);
        perror("fopen");
        return;
    }

    fprintf(output_file, "%.5f,%.5f,%.5f\n", total_merkle_root_time, total_find_nonce_time, total_time);
    fclose(output_file);
}

void generate_blockchain(const char *input_file_name) {
    BYTE difficulty[SHA256_HASH_SIZE], prev_block_hash[SHA256_HASH_SIZE];
    int max_transactions_in_a_block, transaction_size, number_of_transactions;
    uint32_t max_nonce;

    FILE *input_file;
    if (read_input(input_file_name, prev_block_hash, difficulty, &max_nonce,
                   &max_transactions_in_a_block, &transaction_size, &number_of_transactions, &input_file) != 0) {
        fprintf(stderr, "Error at reading input\n");
        perror("fopen");
        return;
    }

    BYTE *transactions = (BYTE *)malloc(max_transactions_in_a_block * transaction_size);
    BYTE merkle_root[SHA256_HASH_SIZE], block_content[BLOCK_SIZE], block_hash[SHA256_HASH_SIZE];
    uint32_t nonce;
    int block_number = 0;
    double start_time, end_time, merkle_root_time, find_nonce_time, total_block_time;
    double total_time = 0, total_merkle_root_time = 0, total_find_nonce_time = 0;

    while (!feof(input_file)) {
        int txs_read = 0;
        memset(transactions, 0, max_transactions_in_a_block * transaction_size);
        read_transactions(transaction_size, transactions, max_transactions_in_a_block, &txs_read, input_file);

        if (txs_read == 0) break;  // End of file reached

        // Construct the Merkle root
        start_time = get_wall_time_sec();
        construct_merkle_root(transaction_size, transactions, max_transactions_in_a_block, txs_read, merkle_root);
        end_time = get_wall_time_sec();
        merkle_root_time = end_time - start_time;
        total_merkle_root_time += merkle_root_time;

        // Prepare block content
        strcpy((char *)block_content, (const char *)prev_block_hash);
        strcat((char *)block_content, (const char *)merkle_root);
        size_t current_length = strlen((char *)block_content);

        // Find nonce
        start_time = get_wall_time_sec();
        int nonce_found = find_nonce(difficulty, max_nonce, block_content, current_length, block_hash, &nonce) == 0;
        end_time = get_wall_time_sec();
        find_nonce_time = end_time - start_time;
        total_find_nonce_time += find_nonce_time;

        total_block_time = merkle_root_time + find_nonce_time;
        total_time += total_block_time;

        // Print results
        print_mining_results(input_file_name, block_number, nonce, block_hash, merkle_root_time, find_nonce_time, total_block_time, nonce_found);

        if (!nonce_found) break;

        // Update previous block hash
        memcpy(prev_block_hash, block_hash, SHA256_HASH_SIZE);
        block_number++;
    }

    free(transactions);
    fclose(input_file);

    print_total_time(input_file_name, total_time, total_merkle_root_time, total_find_nonce_time);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <TEST>\n", argv[0]);
        return 1;
    }

    printf("Generating blockchain for %s\n", argv[1]);
    warm_up_gpu();
    generate_blockchain(argv[1]);
    printf("Finished!\n");

    return 0;
}
