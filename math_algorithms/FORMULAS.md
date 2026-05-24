# Atomic formulas

Let `G = graph`.

Let `V = verifier`.

Let `H = heuristic`.

Let `D = data`.

Let `T = transformer`.

`D* = (G + V + H) -> better labels -> T`

`d[v] = min(d[v], d[u] + w(u,v))`

`f(n) = g(n) + h(n)`

`topological_order(u) < topological_order(v)` for every edge `u -> v`.

`lowlink(v) = min(index(v), index(w), lowlink(w))`

`find(x) = representative(parent[x])`

`rule(head) <- predicate_1, predicate_2, ...`

`F(x) = join(transfer(F(pred_1)), ..., transfer(F(pred_n)))`

`eclass(a) = eclass(b)` when equality saturation proves `a = b`.

`exists x . verifier(candidate(x)) = true`

`beam_{k+1} = top_b(expand(beam_k))`

`property(input) = true` for generated inputs.

`X_k = sum_{n=0}^{N-1} x_n * exp(-2*pi*i*k*n/N)`

`C[i,j] = sum_k A[i,k] * B[k,j]`

`A x = b`

`maximize c^T x subject to A x <= b, x >= 0`

`max_flow(s,t) = min_cut(s,t)`

`visited(v) = true`

`pi[i] = length(longest proper prefix that is also suffix)`

`root = hash(left || right)`

`max(correctness, efficiency, learning, robustness) = good`
