# SparseRNJ evaluation
> Running RNJ/SRNJ on SCONCE2 and MEDICC2 distances

## Data
Data is simulated with CNAsim

## Execution
MEDICC2 requires copy number profiles, therefore the integer
CN from SCONCE2 output is used.

## Evaluation
The performance of a method is evaluated by comparing the inferred tree
to the true tree used for simulation with both the RF distance and the
quartet distance.
Methods that are compared are:
- SCONCE2 + NJ (original)
- SCONCE2 + RNJ (using t1, t2, t3)
- SCONCE2 + SRNJ (using t1, t2, t3, O(n log n) complexity)
- MEDICC2 + NJ (original)
- MEDICC2 + RNJ (using DLCA estimate of t1)
- MEDICC2 + SRNJ (using DLCA estimate of t1, O(n log n) complexity)