# Positive control: dacl10k fasterrcnn, planted class Rust

## O0 manipulation check (K-uniform − C-uniform, test AP50 of the planted class)

-0.29, seed t [-1.62, +1.04], bootstrap [-0.78, +0.26]. Pass: False

## O1 detection (raw utility of the planted edit)

| Arm | Seed | u(plant, c*) | t | mean other classes | c* most negative |
|---|---|---|---|---|---|
| shared | 0 | -0.00232 | -11.53 | -0.00052 | True |
| shared | 1 | -0.00265 | -11.28 | -0.000532 | True |
| shared | 2 | -0.00251 | -10.86 | -0.000532 | True |
| cua | 0 | -0.00227 | -11.51 | -0.000508 | True |
| cua | 1 | -0.00263 | -11.26 | -0.00052 | True |
| cua | 2 | -0.0026 | -10.67 | -0.000539 | True |
| shared2 | 0 | -0.111 | -6.13 | -0.0177 | True |
| shared2 | 1 | -0.124 | -6.83 | -0.024 | True |
| shared2 | 2 | -0.107 | -10.11 | -0.0155 | True |
| cua2 | 0 | -0.109 | -7.73 | -0.0153 | True |
| cua2 | 1 | -0.129 | -7.34 | -0.0207 | True |
| cua2 | 2 | -0.111 | -9.81 | -0.0177 | True |

shared: contrast seed t [-0.00236, -0.00156]. Pass: True

cua: contrast seed t [-0.00244, -0.00151]. Pass: True

shared2: contrast seed t [-0.106, -0.084]. Pass: True

cua2: contrast seed t [-0.119, -0.0774]. Pass: True

## O2 action (planted-edit rate, images with c* / images with other classes only)

Target-free images bypass the policy and are excluded from the denominator group; the last column keeps them in, as in the first read-out.

| Arm | Seed | rate with c* | rate without | ratio | ratio incl. target-free |
|---|---|---|---|---|---|
| uniform | 0 | 0.0665 | 0.0672 | 0.990 | 1.068 |
| uniform | 1 | 0.0666 | 0.0685 | 0.971 | 1.041 |
| uniform | 2 | 0.0670 | 0.0678 | 0.988 | 1.062 |
| shared | 0 | 0.0003 | 0.0003 | 0.978 | 0.060 |
| shared | 1 | 0.0002 | 0.0002 | 1.293 | 0.039 |
| shared | 2 | 0.0013 | 0.0014 | 0.929 | 0.189 |
| cua | 0 | 0.0000 | 0.0004 | 0.131 | 0.009 |
| cua | 1 | 0.0000 | 0.0004 | 0.113 | 0.008 |
| cua | 2 | 0.0008 | 0.0013 | 0.598 | 0.116 |
| shared2 | 0 | 0.0000 | 0.0000 | n/a | 0.000 |
| shared2 | 1 | 0.0000 | 0.0000 | n/a | 0.000 |
| shared2 | 2 | 0.0000 | 0.0000 | n/a | 0.000 |
| cua2 | 0 | 0.0000 | 0.0016 | 0.000 | 0.000 |
| cua2 | 1 | 0.0000 | 0.0077 | 0.000 | 0.000 |
| cua2 | 2 | 0.0000 | 0.0015 | 0.000 | 0.000 |

uniform: mean ratio 0.983. Pass (all ≤ 0.33): False
shared: mean ratio 1.067. Pass (all ≤ 0.33): False
cua: mean ratio 0.281. Pass (all ≤ 0.33): False
shared2: mean ratio n/a (no planted edit on any image the policy decides). Pass (all ≤ 0.33): False
cua2: mean ratio 0.000. Pass (all ≤ 0.33): True

## O3 benefit (arm − K-uniform, test AP50 of c*)

- shared: -0.88, seed t [-2.22, +0.45], bootstrap [-1.86, +0.25]. Pass: False
- cua: -0.90, seed t [-1.96, +0.15], bootstrap [-1.95, +0.17]. Pass: False
- shared2: +0.38, seed t [+0.04, +0.73], bootstrap [-0.19, +0.92]. Pass: False
- cua2: +0.12, seed t [-1.29, +1.53], bootstrap [-0.53, +0.86]. Pass: False
