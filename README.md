# lqp
Learning Quadratic Programs with Neural Networks

See our paper:
https://arxiv.org/pdf/2112.07464.pdf

Dependencies:
torch >= v0.6.0
quadprog >= 1.5.8
scs >= 3.0.0

## Currently Supports:
### Linear programming:
1. Alternating direction method of multiplier (ADMM)

### Quadratic programming:
1. Unconstrained solver
2. Linear equality constrained solver
3. Alternating direction method of multiplier (ADMM)
4. Primal-dual interior point solver (in python see qpth/OptNet)
5. Active set solver (see quadprog).

### Coming soon:

### Quadratic programming:
1. Operator splitting quadratic programming (OSQP)

