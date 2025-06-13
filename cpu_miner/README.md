- Directorul cpu_miner conține implementarea serială, pe CPU. Acesta nu face parte din rezolvarea temei. Scopul codului este de a înțelege funcționalitatea pe CPU, ca apoi să optimizați căutarea nonce-ului si generarea Merkle root-ului pe GPU, folosind CUDA.
- Aceasta este o abordare simplistă a calculării unui block hash, cu complexitate redusă. Nu reflectă implementarea reală a algoritmilor de consens POW, având scop pur educativ.

Pasi pentru rulare:

1. To compile: make
2. To run: make run TEST=<TEST_NAME>. Example: make run TEST=test1
3. To clean: make clean
