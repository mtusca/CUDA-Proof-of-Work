- În directorul gpu_miner veți realiza paralelizarea în CUDA a logicii funcțiilor construct_merkle_root si find_nonce din cpu_miner.
- Pentru a vă ajută, aveți deja implementate funcții ajutătoare în utils.cu. Vă recomandăm să va folosiți de ele în implementarea voastră.

Observații:

1. Lucrați DOAR în fișierul utils.cu. La încărcarea pe Moodle, veți încărca DOAR fișierul utils.cu și un fișier README.md cu descrierea temei și eventuale observații. Restul fișierelor vor fi ignorate, fiind suprascrise la testarea automată a temei!
2. Aveți la dispoziție 4 fișiere de intrare pentru testare. Sunteți încurajați să vă creați propriile teste pentru a verifica corectitudinea și performanța implementării voastre.
3. Compilarea și rularea temei se vor face EXCLUSIV pe coada xl (testarea automată va fi, de asemenea, făcută pe xl). Nu este nevoie să modificați nimic în Makefile, regulile sunt deja făcute. Tot ce trebuie să faceți este să apelați make, make run și make clean, ca la laborator.

Pași pentru rulare:

1. To compile: make
2. To run: make run TEST=<test_name>
3. To clean: make clean
