/**
 * optimizers.js
 * Implementations of 7 gradient-based optimizers.
 * Each function takes (grad, x0start, x1start, lr, numIter, ...hyperparams)
 * and returns a path: Array of [x0, x1] pairs.
 */

'use strict';

/**
 * SGD — Vanilla gradient descent
 *   x ← x − α·∇f(x)
 */
function runSGD(grad, x0s, x1s, lr, numIter) {
  let x = [x0s, x1s];
  const path = [[...x]];
  for (let t = 0; t < numIter; t++) {
    let g;
    try { g = grad(x[0], x[1]); } catch (e) { break; }
    if (!isFinite(g[0]) || !isFinite(g[1])) break;
    x = [x[0] - lr * g[0], x[1] - lr * g[1]];
    if (!isFinite(x[0]) || !isFinite(x[1])) { x = path[path.length - 1]; break; }
    path.push([...x]);
  }
  return path;
}

/**
 * Momentum — SGD with velocity accumulation
 *   v ← β·v + α·∇f(x)
 *   x ← x − v
 */
function runMomentum(grad, x0s, x1s, lr, numIter, beta) {
  let x = [x0s, x1s], v = [0, 0];
  const path = [[...x]];
  for (let t = 0; t < numIter; t++) {
    let g;
    try { g = grad(x[0], x[1]); } catch (e) { break; }
    if (!isFinite(g[0]) || !isFinite(g[1])) break;
    v = [beta * v[0] + lr * g[0], beta * v[1] + lr * g[1]];
    x = [x[0] - v[0], x[1] - v[1]];
    if (!isFinite(x[0]) || !isFinite(x[1])) { x = path[path.length - 1]; break; }
    path.push([...x]);
  }
  return path;
}

/**
 * Nesterov Accelerated Gradient (NAG)
 *   x̃ ← x − β·v          (lookahead position)
 *   v ← β·v + α·∇f(x̃)   (update velocity at lookahead)
 *   x ← x − v
 */
function runNesterov(grad, x0s, x1s, lr, numIter, beta) {
  let x = [x0s, x1s], v = [0, 0];
  const path = [[...x]];
  for (let t = 0; t < numIter; t++) {
    const xLook = [x[0] - beta * v[0], x[1] - beta * v[1]];
    let g;
    try { g = grad(xLook[0], xLook[1]); } catch (e) { break; }
    if (!isFinite(g[0]) || !isFinite(g[1])) break;
    v = [beta * v[0] + lr * g[0], beta * v[1] + lr * g[1]];
    x = [x[0] - v[0], x[1] - v[1]];
    if (!isFinite(x[0]) || !isFinite(x[1])) { x = path[path.length - 1]; break; }
    path.push([...x]);
  }
  return path;
}

/**
 * AdaGrad — adaptive learning rates via accumulated squared gradients
 *   G  ← G + g²
 *   x  ← x − (α / √(G + ε)) · g
 */
function runAdaGrad(grad, x0s, x1s, lr, numIter, eps) {
  let x = [x0s, x1s], G = [0, 0];
  const path = [[...x]];
  for (let t = 0; t < numIter; t++) {
    let g;
    try { g = grad(x[0], x[1]); } catch (e) { break; }
    if (!isFinite(g[0]) || !isFinite(g[1])) break;
    G = [G[0] + g[0] ** 2, G[1] + g[1] ** 2];
    x = [
      x[0] - lr * g[0] / (Math.sqrt(G[0] + eps)),
      x[1] - lr * g[1] / (Math.sqrt(G[1] + eps)),
    ];
    if (!isFinite(x[0]) || !isFinite(x[1])) { x = path[path.length - 1]; break; }
    path.push([...x]);
  }
  return path;
}

/**
 * RMSProp — fixes AdaGrad's vanishing lr via exponential moving average
 *   E[g²] ← ρ·E[g²] + (1−ρ)·g²
 *   x     ← x − (α / √(E[g²] + ε)) · g
 */
function runRMSProp(grad, x0s, x1s, lr, numIter, rho, eps) {
  let x = [x0s, x1s], E = [0, 0];
  const path = [[...x]];
  for (let t = 0; t < numIter; t++) {
    let g;
    try { g = grad(x[0], x[1]); } catch (e) { break; }
    if (!isFinite(g[0]) || !isFinite(g[1])) break;
    E = [rho * E[0] + (1 - rho) * g[0] ** 2, rho * E[1] + (1 - rho) * g[1] ** 2];
    x = [
      x[0] - lr * g[0] / (Math.sqrt(E[0] + eps)),
      x[1] - lr * g[1] / (Math.sqrt(E[1] + eps)),
    ];
    if (!isFinite(x[0]) || !isFinite(x[1])) { x = path[path.length - 1]; break; }
    path.push([...x]);
  }
  return path;
}

/**
 * Adam — Adaptive Moment Estimation
 *   m   ← β₁·m + (1−β₁)·g           (1st moment / mean)
 *   v   ← β₂·v + (1−β₂)·g²          (2nd moment / uncentered variance)
 *   m̂  = m / (1 − β₁ᵗ)              (bias-corrected 1st moment)
 *   v̂  = v / (1 − β₂ᵗ)              (bias-corrected 2nd moment)
 *   x   ← x − α · m̂ / (√v̂ + ε)
 */
function runAdam(grad, x0s, x1s, lr, numIter, beta1, beta2, eps) {
  let x = [x0s, x1s], m = [0, 0], v = [0, 0];
  const path = [[...x]];
  for (let t = 1; t <= numIter; t++) {
    let g;
    try { g = grad(x[0], x[1]); } catch (e) { break; }
    if (!isFinite(g[0]) || !isFinite(g[1])) break;
    m = [beta1 * m[0] + (1 - beta1) * g[0], beta1 * m[1] + (1 - beta1) * g[1]];
    v = [beta2 * v[0] + (1 - beta2) * g[0] ** 2, beta2 * v[1] + (1 - beta2) * g[1] ** 2];
    const mc = [m[0] / (1 - beta1 ** t), m[1] / (1 - beta1 ** t)];
    const vc = [v[0] / (1 - beta2 ** t), v[1] / (1 - beta2 ** t)];
    x = [
      x[0] - lr * mc[0] / (Math.sqrt(vc[0]) + eps),
      x[1] - lr * mc[1] / (Math.sqrt(vc[1]) + eps),
    ];
    if (!isFinite(x[0]) || !isFinite(x[1])) { x = path[path.length - 1]; break; }
    path.push([...x]);
  }
  return path;
}

/**
 * AdamW — Adam with decoupled weight decay (Loshchilov & Hutter 2019)
 *   Weight decay is applied DIRECTLY to parameters, not folded into the gradient.
 *   This is the correct way to do L2 regularization with adaptive optimizers.
 *
 *   x ← x·(1 − α·λ) − α · m̂ / (√v̂ + ε)
 */
function runAdamW(grad, x0s, x1s, lr, numIter, beta1, beta2, eps, wd) {
  let x = [x0s, x1s], m = [0, 0], v = [0, 0];
  const path = [[...x]];
  for (let t = 1; t <= numIter; t++) {
    let g;
    try { g = grad(x[0], x[1]); } catch (e) { break; }
    if (!isFinite(g[0]) || !isFinite(g[1])) break;
    m = [beta1 * m[0] + (1 - beta1) * g[0], beta1 * m[1] + (1 - beta1) * g[1]];
    v = [beta2 * v[0] + (1 - beta2) * g[0] ** 2, beta2 * v[1] + (1 - beta2) * g[1] ** 2];
    const mc = [m[0] / (1 - beta1 ** t), m[1] / (1 - beta1 ** t)];
    const vc = [v[0] / (1 - beta2 ** t), v[1] / (1 - beta2 ** t)];
    // Decoupled weight decay applied before the adaptive step
    x = [
      x[0] * (1 - lr * wd) - lr * mc[0] / (Math.sqrt(vc[0]) + eps),
      x[1] * (1 - lr * wd) - lr * mc[1] / (Math.sqrt(vc[1]) + eps),
    ];
    if (!isFinite(x[0]) || !isFinite(x[1])) { x = path[path.length - 1]; break; }
    path.push([...x]);
  }
  return path;
}

/**
 * Dispatch helper — routes to correct optimizer given its id and DOM params.
 */
function dispatchOptimizer(optId, grad, x0s, x1s, lr, numIter) {
  const p = (pid) => {
    const el = document.getElementById(`p-${optId}-${pid}`);
    return el ? parseFloat(el.value) : null;
  };
  switch (optId) {
    case 'sgd':      return runSGD(grad, x0s, x1s, lr, numIter);
    case 'momentum': return runMomentum(grad, x0s, x1s, lr, numIter, p('beta'));
    case 'nesterov': return runNesterov(grad, x0s, x1s, lr, numIter, p('beta'));
    case 'adagrad':  return runAdaGrad(grad, x0s, x1s, lr, numIter, p('eps') * 1e-8);
    case 'rmsprop':  return runRMSProp(grad, x0s, x1s, lr, numIter, p('rho'), p('eps') * 1e-8);
    case 'adam':     return runAdam(grad, x0s, x1s, lr, numIter, p('beta1'), p('beta2'), p('eps') * 1e-8);
    case 'adamw':    return runAdamW(grad, x0s, x1s, lr, numIter, p('beta1'), p('beta2'), p('eps') * 1e-8, p('wd'));
    default:         return [];
  }
}

/**
 * Numerical gradient via central differences.
 */
function numericalGrad(f, x0, x1, h = 1e-5) {
  return [
    (f(x0 + h, x1, Math) - f(x0 - h, x1, Math)) / (2 * h),
    (f(x0, x1 + h, Math) - f(x0, x1 - h, Math)) / (2 * h),
  ];
}

/**
 * Find global minimum via grid search + Adam refinement.
 * Returns { x, y, z } or null if the function appears unbounded.
 */
function findGlobalMin(f, gradFn) {
  const N = 30, r = 60;   // search radius matches expanded ±50 start range + buffer
  let bv = Infinity, bx = [0, 0];
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      const x0 = -r + 2 * r * i / N;
      const x1 = -r + 2 * r * j / N;
      try {
        const z = f(x0, x1, Math);
        if (isFinite(z) && z < bv) { bv = z; bx = [x0, x1]; }
      } catch (e) {}
    }
  }
  // Refine from best candidate using Adam
  try {
    const rp = runAdam(gradFn, bx[0], bx[1], 0.05, 3000, 0.9, 0.999, 1e-8);
    const rx = rp[rp.length - 1];
    const rz = f(rx[0], rx[1], Math);
    if (isFinite(rz) && rz < bv) { bv = rz; bx = rx; }
  } catch (e) {}

  // If no valid grid point was found, we can't determine a minimum
  if (!isFinite(bv)) return null;

  // Unbounded check: if any boundary point is lower, no bounded min exists
  const boundary = [[-60,-60],[-60,60],[60,-60],[60,60],[0,-60],[0,60],[-60,0],[60,0]];
  for (const [cx, cy] of boundary) {
    try {
      const cz = f(cx, cy, Math);
      if (isFinite(cz) && cz < bv - 0.1) return null;
    } catch (e) {}
  }
  return { x: bx[0], y: bx[1], z: bv };
}
