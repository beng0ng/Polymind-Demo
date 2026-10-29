/* Opponents for the browser game: the trained network and the scripted expert.
 *
 * PolicyNet mirrors polytopia/rl/model.py (residual CNN, per-tile action head)
 * and runs on weights exported by scripts/export_web.py. Expert mirrors
 * polytopia/agents/heuristic.py.
 */
(function (root) {
  "use strict";
  const Poly = root.Poly || (typeof require !== "undefined" ? require("./engine.js") : null);
  const { AK, UA, TA, cheb, combatDamage } = Poly;

  class PolicyNet {
    constructor(meta, buffer) {
      this.meta = meta;
      const all = new Float32Array(buffer);
      this.w = {};
      for (const [name, { offset, shape }] of Object.entries(meta.tensors)) {
        const n = shape.reduce((a, b) => a * b, 1);
        this.w[name] = { data: all.subarray(offset, offset + n), shape };
      }
      this.blocks = meta.blocks;
    }
    conv(x, cin, h, w, name, k) {
      // im2col: gather the k*k shifted copies of every input plane once, then each
      // output plane is a weighted sum of those rows (same arithmetic, tighter loops).
      const W = this.w[name + ".weight"], B = this.w[name + ".bias"];
      const cout = W.shape[0], HW = h * w, out = new Float32Array(cout * HW), wd = W.data, pad = (k - 1) >> 1;
      const K = cin * k * k;
      let col = this._col;
      if (!col || col.length < K * HW) col = this._col = new Float32Array(K * HW);
      else col.fill(0, 0, K * HW);
      for (let ci = 0; ci < cin; ci++) {
        const inp = x.subarray(ci * HW, (ci + 1) * HW);
        for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
          const row = col.subarray(((ci * k + ky) * k + kx) * HW, ((ci * k + ky) * k + kx + 1) * HW);
          const dy = ky - pad, dx = kx - pad;
          const y0 = Math.max(0, -dy), y1 = Math.min(h, h - dy), x0 = Math.max(0, -dx), x1 = Math.min(w, w - dx);
          for (let y = y0; y < y1; y++) {
            const ro = y * w, ri = (y + dy) * w + dx;
            for (let xx = x0; xx < x1; xx++) row[ro + xx] = inp[ri + xx];
          }
        }
      }
      for (let co = 0; co < cout; co++) {
        const o = out.subarray(co * HW, (co + 1) * HW), wo = co * K;
        o.fill(B.data[co]);
        for (let q = 0; q < K; q++) {
          const wv = wd[wo + q];
          if (wv === 0) continue;
          const base = q * HW;
          for (let i = 0; i < HW; i++) o[i] += wv * col[base + i];
        }
      }
      return out;
    }
    linear(v, name) {
      const W = this.w[name + ".weight"], B = this.w[name + ".bias"], [o, i] = W.shape, out = new Float32Array(o);
      for (let r = 0; r < o; r++) {
        let s = B.data[r];
        const row = r * i;
        for (let c = 0; c < i; c++) s += W.data[row + c] * v[c];
        out[r] = s;
      }
      return out;
    }
    forward(board, globals, h, w) {
      const relu = (a) => { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; };
      const cin = this.w["stem.weight"].shape[1], ch = this.w["stem.weight"].shape[0], HW = h * w;
      let x = relu(this.conv(board, cin, h, w, "stem", 3));
      for (let b = 0; b < this.blocks; b++) {
        const y = this.conv(relu(this.conv(x, ch, h, w, `trunk.${b}.c1`, 3)), ch, h, w, `trunk.${b}.c2`, 3);
        for (let i = 0; i < x.length; i++) y[i] += x[i];
        x = relu(y);
      }
      const tile = this.conv(x, ch, h, w, "tile_head", 1);
      const per = this.w["tile_head.weight"].shape[0];
      const pooled = new Float32Array(2 * ch + globals.length);
      for (let c = 0; c < ch; c++) {
        let s = 0, m = -Infinity;
        for (let i = 0; i < HW; i++) { const v = x[c * HW + i]; s += v; if (v > m) m = v; }
        pooled[c] = s / HW;
        pooled[ch + c] = m;
      }
      pooled.set(globals, 2 * ch);
      const hdn = relu(this.linear(relu(this.linear(pooled, "global_mlp.0")), "global_mlp.2"));
      const glob = this.linear(hdn, "global_head");
      const value = this.linear(hdn, "value_head")[0];
      const win = this.w["win_head.0.weight"] ? Math.tanh(this.linear(relu(this.linear(hdn, "win_head.0")), "win_head.2")[0]) : undefined;
      const logits = new Float32Array(glob.length + HW * per);
      logits.set(glob, 0);
      for (let i = 0; i < HW; i++) for (let k = 0; k < per; k++) logits[glob.length + i * per + k] = tile[k * HW + i];
      return { logits, value, win };
    }
  }

  // Adds a win-probability head (polytopia/rl/model.py WinHead, exported by
  // scripts/export_web.py --win-head) to a PolicyNet. forward() then also returns
  // `win`, a win estimate in [-1, 1]; the policy and the value head are unchanged.
  function attachWinHead(net, spec) {
    const dec = (b64) => {
      const bin = typeof atob === "function" ? Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)) : Buffer.from(b64, "base64");
      return new Float32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
    };
    for (const [name, { shape, data }] of Object.entries(spec.tensors)) net.w[name] = { data: dec(data), shape };
    return net;
  }

  // Same network on the GPU through TensorFlow.js (WebGL), when the page has it.
  // Weights are the PolicyNet's, re-laid out for NHWC; forward() returns the same
  // logits and value (to float rounding), so callers can use either.
  class TfNet {
    constructor(cpuNet) {
      const tf = root.tf, w = cpuNet.w;
      this.tf = tf; this.blocks = cpuNet.blocks; this.cpu = cpuNet; this.meta = cpuNet.meta; this.w = cpuNet.w;
      const convW = (n) => { const [o, i, kh, kw] = w[n + ".weight"].shape;
        return tf.tidy(() => tf.tensor4d(w[n + ".weight"].data, [o, i, kh, kw]).transpose([2, 3, 1, 0])); };
      const linW = (n) => { const [o, i] = w[n + ".weight"].shape;
        return tf.tidy(() => tf.tensor2d(w[n + ".weight"].data, [o, i]).transpose()); };
      const bias = (n) => tf.tensor1d(w[n + ".bias"].data);
      this.p = {};
      const convs = ["stem", "tile_head"];
      for (let b = 0; b < this.blocks; b++) convs.push(`trunk.${b}.c1`, `trunk.${b}.c2`);
      for (const n of convs) this.p[n] = [convW(n), bias(n)];
      for (const n of ["global_mlp.0", "global_mlp.2", "global_head", "value_head"]) this.p[n] = [linW(n), bias(n)];
      this.hasWin = !!w["win_head.0.weight"];
      if (this.hasWin) for (const n of ["win_head.0", "win_head.2"]) this.p[n] = [linW(n), bias(n)];
    }
    forward(board, globals, h, w) { return this.forwardBatch([board], [globals], h, w)[0]; }
    forwardBatch(boards, globs, h, w) {
      const tf = this.tf, P = this.p, cin = this.cpu.w["stem.weight"].shape[1], N = boards.length, G = globs[0].length;
      const bb = new Float32Array(N * cin * h * w), gg = new Float32Array(N * G);
      boards.forEach((b, i) => bb.set(b, i * b.length));
      globs.forEach((g, i) => gg.set(g, i * G));
      const [lg, val] = tf.tidy(() => {
        const conv = (x, n) => tf.add(tf.conv2d(x, P[n][0], 1, "same"), P[n][1]);
        let x = tf.tensor4d(bb, [N, cin, h, w]).transpose([0, 2, 3, 1]);
        x = tf.relu(conv(x, "stem"));
        for (let b = 0; b < this.blocks; b++) x = tf.relu(tf.add(conv(tf.relu(conv(x, `trunk.${b}.c1`)), `trunk.${b}.c2`), x));
        const tile = conv(x, "tile_head").reshape([N, -1]);
        const pooled = tf.concat([tf.mean(x, [1, 2]), tf.max(x, [1, 2]), tf.tensor2d(gg, [N, G])], 1);
        const lin = (v, n) => tf.add(tf.matMul(v, P[n][0]), P[n][1]);
        const hdn = tf.relu(lin(tf.relu(lin(pooled, "global_mlp.0")), "global_mlp.2"));
        const v = lin(hdn, "value_head");
        return [tf.concat([lin(hdn, "global_head"), tile], 1), this.hasWin ? tf.concat([v, tf.tanh(lin(tf.relu(lin(hdn, "win_head.0")), "win_head.2"))], 1) : v];
      });
      const L = lg.dataSync(), V = val.dataSync(), A = L.length / N, nv = V.length / N;
      lg.dispose(); val.dispose();
      return boards.map((_, i) => ({ logits: Float32Array.from(L.subarray(i * A, (i + 1) * A)), value: V[i * nv],
                                     win: nv > 1 ? V[i * nv + 1] : undefined }));
    }
  }
  // Uses the GPU network when WebGL is available, the plain JS one otherwise.
  async function fastNet(cpuNet) {
    try {
      if (!root.tf) return cpuNet;
      await root.tf.setBackend("webgl"); await root.tf.ready();
      if (root.tf.getBackend() !== "webgl") return cpuNet;
      const g = new TfNet(cpuNet);
      g.forward(new Float32Array(cpuNet.w["stem.weight"].shape[1] * 121), new Float32Array(cpuNet.w["global_mlp.0.weight"].shape[1] - 2 * cpuNet.w["stem.weight"].shape[0]), 11, 11);  // warm up
      return g;
    } catch (e) { return cpuNet; }
  }

  PolicyNet.prototype.forwardBatch = function (boards, globs, h, w) {
    return boards.map((b, i) => this.forward(b, globs[i], h, w));
  };

  function sampleMasked(logits, legalIdx, rand, greedy) {
    let m = -Infinity;
    for (const i of legalIdx) if (logits[i] > m) m = logits[i];
    if (greedy) return legalIdx.find((i) => logits[i] === m);
    const p = legalIdx.map((i) => Math.exp(logits[i] - m));
    let s = p.reduce((a, b) => a + b, 0), r = rand() * s;
    for (let k = 0; k < p.length; k++) { r -= p[k]; if (r <= 0) return legalIdx[k]; }
    return legalIdx[legalIdx.length - 1];
  }

  // Picks one action for the side to move. `actionsThisTurn` enforces the
  // training env's cap of 250 actions per turn.
  function netAction(net, game, actionsThisTurn, rand = Math.random) {
    const legal = game.legalActions();
    const p = game.players[game.current];
    if (actionsThisTurn >= 250 && !p.pending.length) return legal.find((a) => a.kind === AK.END_TURN);
    const codec = Poly.codec(game.R), byIdx = new Map();
    for (const a of legal) byIdx.set(codec.encode(a, game.width), a);
    const obs = Poly.encodeObservation(game, game.current);
    const { logits } = net.forward(obs.board, obs.globals, game.height, game.width);
    return byIdx.get(sampleMasked(logits, [...byIdx.keys()], rand, false));
  }

  // ---------------------------------------------------------------- expert
  const REWARD_PREF = { 0: 2, 1: 1, 3: 2, 2: 1, 4: 2, 5: 1, 7: 2, 6: 1 };
  const TECH_ORDER = [10, 15, 20, 0, 12, 11, 16, 1, 17, 21, 3, 22, 14, 24, 18, 5, 2, 4];
  const UNIT_PREF = { 6: 9, 4: 8, 5: 6, 2: 6, 1: 5, 3: 4, 0: 3, 8: 2 };

  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function expertAction(game, noise = 0.5) {
    const pid = game.current, p = game.players[pid], ex = p.explored, W = game.width;
    const targets = game.cities.filter((c) => c.owner !== pid && ex[c.y * W + c.x]).map((c) => [c.x, c.y]);
    const ruins = [], unexplored = [];
    for (let y = 0; y < game.height; y++) for (let x = 0; x < W; x++) {
      if (game.tile(x, y).ruin && ex[y * W + x]) ruins.push([x, y]);
      if (!ex[y * W + x]) unexplored.push([x, y]);
    }
    const enemies = [...game.units.values()].filter((u) => u.owner !== pid && game.seesUnit(pid, u)).map((u) => [u.x, u.y]);
    const nUnits = game.playerUnits(pid).length, nCities = game.playerCities(pid).length;
    const nearest = (x, y, pts) => pts.reduce((m, [a, b]) => Math.min(m, cheb(x, y, a, b)), 99);
    // Threats to our cities, Giants in sight, and what the enemy's score says.
    const threatened = new Map();
    for (const c of game.playerCities(pid)) {
      const [t, d] = game.cityThreat(c);
      if (t > d) threatened.set(c.y * W + c.x, [c.x, c.y]);
    }
    let giants = 0, army = 0, teching = false, growing = false;
    for (const q of game.players) {
      if (!q.alive || q.id === pid) continue; // every tribe's score is public (scoreboard), met or not
      const [a, g] = game.visibleArmy(pid, q.id);
      army += a; giants += g;
      const hidden = game.lastHiddenGain[pid + ":" + q.id] ?? 0;
      teching = teching || hidden >= 100;
      growing = growing || (game.lastGain[q.id] ?? 0) - hidden >= 25;
    }
    const mine = game.playerUnits(pid).reduce((m, u) => m + game.R.units[u.type].cost, 0);
    const outgunned = army > mine;

    function score(a) {
      switch (a.kind) {
        case AK.END_TURN: return 0;
        case AK.REWARD: return REWARD_PREF[p.pending[0][1][a.arg]];
        case AK.RESEARCH: {
          if (a.arg === 11 && giants) return 650; // Archery: ranged units counter Giants
          const r = TECH_ORDER.indexOf(a.arg); return 420 - 12 * (r < 0 ? TECH_ORDER.length : r);
        }
        case AK.UNIT: {
          const u = game.unitOn(a.x, a.y);
          if (a.arg === UA.CAPTURE) return 1000;
          if (a.arg === UA.EXAMINE) return 900;
          if (a.arg === UA.HEAL_OTHERS) return 300;
          if (a.arg === UA.RECOVER) return u.hp < u.maxHp / 2 ? 250 : 30;
          if (a.arg === UA.UPGRADE) return 120;
          if (a.arg === UA.PROMOTE) return u.hp < u.maxHp * 0.6 ? 850 : 20;  // the full heal is worth most when wounded
          return -100;
        }
        case AK.ATTACK: {
          const at = game.unitOn(a.x, a.y), df = game.unitOn(a.tx, a.ty);
          if (at.type === 9) return 750;  // Cloak infiltrating a city
          const cost = game.R.units[df.type].cost;
          if (at.type === 8 && !at.naval) return 700 + cost * 10;
          const [d, r] = combatDamage(game.attackOf(at), at.hp, at.maxHp, game.defenseOf(df), df.hp, df.maxHp, game.defenseBonus(df), game.postHit);
          const nearHome = [...threatened.values()].some(([x, y]) => cheb(df.x, df.y, x, y) <= 2);
          const focus = nearHome ? 150 : (game.rangeOf(df) > 1 && !threatened.size ? 60 : 0);
          if (d >= df.hp) return 800 + cost * 10 + focus;
          if (game.exposedFirstStrike(at, df)) return -50;
          const inRange = cheb(at.x, at.y, df.x, df.y) <= game.rangeOf(df);
          return 300 + 10 * d - (inRange ? 8 * r : 0) + focus;
        }
        case AK.TILE: {
          const s = game.R.tileActions[a.arg];
          if (a.arg === TA.ROAD || a.arg === TA.BRIDGE) return 5;
          if (a.arg === TA.GROW_FOREST || a.arg === TA.BURN_FOREST) return 10;
          if (a.arg === TA.CLEAR_FOREST) return 20;
          if (a.arg === TA.WHALING) return 600;
          return 500 + 60 * s.population - 8 * Math.max(s.cost, 0);
        }
        case AK.TRAIN: {
          let base = nUnits < nCities * 2 + 1 ? 350 : 40;
          if (threatened.has(a.y * W + a.x)) base = 700;
          else if (outgunned || growing) base += 120;
          if (giants && (a.arg === 2 || a.arg === 5)) base += 200;
          return base + 5 * (UNIT_PREF[a.arg] || 1);
        }
        case AK.MOVE: {
          const u = game.unitOn(a.x, a.y), t = game.tile(a.tx, a.ty);
          if (t.cityId !== null && game.cities[t.cityId].owner !== pid) return 600;
          if (t.ruin) return 550;
          let goals = targets.concat(ruins);
          if (!goals.length) goals = enemies.length ? enemies : unexplored;
          if (!goals.length) return -1;
          const here = game.tile(a.x, a.y);
          if (here.cityId !== null && game.cities[here.cityId].owner === pid && nUnits <= 1) return -1;
          const gain = nearest(a.x, a.y, goals) - nearest(a.tx, a.ty, goals);
          if (gain <= 0) return -1;
          const danger = nearest(a.tx, a.ty, enemies) <= 1 && u.hp < u.maxHp / 2;
          const press = teching && !outgunned ? 30 : 0;
          return 150 + 25 * gain + press - (danger ? 100 : 0);
        }
      }
      return 0;
    }
    const legal = game.legalActions();
    let best = null, bs = -Infinity;
    for (const a of legal) {
      const s = score(a) + gauss() * noise * 10;
      if (s > bs) { bs = s; best = a; }
    }
    return best;
  }

  // ---------------------------------------------------------------- search
  // Policy-guided lookahead (mirrors polytopia/agents/search.py, "Search-16" = k 4, m 4).
  // At a tactical decision the policy's top k moves are each tried m times on a
  // copy of the game holding only what this player knows (fogged and hidden enemy
  // units removed, its own random stream), playing the rest of its own turn with
  // the policy and scoring the result with the value head. Never looks into the
  // enemy's turn or past a ruin.
  function deepCopy(v, seen) {
    if (v === null || typeof v !== "object") return v;
    if (seen.has(v)) return seen.get(v);
    let out;
    if (v instanceof Map) { out = new Map(); seen.set(v, out); for (const [k, x] of v) out.set(k, deepCopy(x, seen)); return out; }
    if (v instanceof Set) { out = new Set(); seen.set(v, out); for (const x of v) out.add(deepCopy(x, seen)); return out; }
    if (ArrayBuffer.isView(v)) { out = v.slice(); seen.set(v, out); return out; }
    if (Array.isArray(v)) { out = new Array(v.length); seen.set(v, out); for (let i = 0; i < v.length; i++) out[i] = deepCopy(v[i], seen); return out; }
    out = Object.create(Object.getPrototypeOf(v)); seen.set(v, out);
    for (const k of Object.keys(v)) out[k] = deepCopy(v[k], seen);
    return out;
  }
  const SHARED = new Set(["R", "rules", "techTier", "techPre"]);
  function cloneGame(game, seed) {
    const g = Object.create(Object.getPrototypeOf(game)), seen = new Map();
    for (const k of Object.keys(game)) {
      if (SHARED.has(k) || typeof game[k] === "function") g[k] = game[k];
      else if (k === "history") g[k] = [];
      else g[k] = deepCopy(game[k], seen);
    }
    let a = seed >>> 0;  // independent random stream for the copy
    g.rand = function (n) {
      a = (a + 0x6d2b79f5) >>> 0; let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return Math.floor((((t ^ (t >>> 14)) >>> 0) / 4294967296) * n);
    };
    return g;
  }

  // useWin: score with the win head (see attachWinHead) instead of the value head.
  function policies(net, games, pid, useWin = false) {
    const obs = games.map((g) => Poly.encodeObservation(g, pid));
    const res = net.forwardBatch(obs.map((o) => o.board), obs.map((o) => o.globals), games[0].height, games[0].width);
    return games.map((g, i) => finish(g, res[i], useWin));
  }
  function policy(net, game, pid, useWin = false) { return policies(net, [game], pid, useWin)[0]; }
  function finish(game, { logits, value, win }, useWin) {
    if (useWin) value = win;
    const codec = Poly.codec(game.R), legal = game.legalActions(), byIdx = new Map();
    for (const a of legal) byIdx.set(codec.encode(a, game.width), a);
    const idx = [...byIdx.keys()].sort((a, b) => a - b);  // same order as the Python legal mask
    let m = -Infinity;
    for (const i of idx) if (logits[i] > m) m = logits[i];
    const p = idx.map((i) => Math.exp(logits[i] - m)), s = p.reduce((x, y) => x + y, 0);
    return { idx, p: p.map((x) => x / s), value, byIdx };
  }
  function pick(p, rand) {
    let r = rand(), k = 0;
    for (; k < p.length - 1; k++) { r -= p[k]; if (r <= 0) break; }
    return k;
  }

  class SearchAgent {
    // gate: "critical" (Alpha1+) searches only tactical decisions; "wide" searches every
    // decision where the policy's top move is below `confident`. win: score positions with
    // the win head (Legion0 = wide gate + win head).
    constructor(net, { k = 4, m = 4, maxActions = 40, confident = 0.85, margin = 0.03, rand = Math.random,
                       gate = "critical", win = false } = {}) {
      if (gate !== "critical" && gate !== "wide") throw new Error(`unknown gate ${gate}`);
      if (win && !net.w["win_head.0.weight"]) throw new Error("win = true needs a network with a win head");
      Object.assign(this, { net, k, m, maxActions, confident, margin, rand, gate, win });
      this.stats = { decisions: 0, searched: 0, changed: 0, ms: 0 };
    }
    critical(game, pid, acts) {
      const attacks = acts.filter((a) => a.kind === AK.ATTACK);
      if (attacks.length >= 2) return true;
      if (attacks.some((a) => { const u = game.unitOn(a.x, a.y); return u && u.type === 6; })) return true;  // Knight
      const capture = acts.some((a) => a.kind === AK.UNIT && a.arg === UA.CAPTURE);
      return capture && game.visibleArmy(pid, 1 - pid)[0] > 0;
    }
    knownCopy(game, pid) {
      const g = cloneGame(game, Math.floor(this.rand() * 4294967296)), ex = g.players[pid].explored;
      for (const u of [...g.units.values()]) if (u.owner !== pid && (!ex[g.idx(u.x, u.y)] || u.hidden)) g._removeUnit(u);
      return g;
    }
    continueTurn(g, pid, greedy) {
      const end = () => (g.result.winner === pid ? 1 : -1);
      for (let n = 0; n < this.maxActions; n++) {
        if (g.done) return end();
        if (g.current !== pid) break;
        const { idx, p, value, byIdx } = policy(this.net, g, pid, this.win);
        const j = greedy ? p.indexOf(Math.max(...p)) : pick(p, this.rand);
        const a = byIdx.get(idx[j]);
        if (a.kind === AK.END_TURN || (a.kind === AK.UNIT && a.arg === UA.EXAMINE)) return value;
        g.apply(a);
      }
      if (g.done) return end();
      return policy(this.net, g, pid, this.win).value;
    }
    act(game, actionsThisTurn = 0) {
      const pid = game.current, pl = game.players[pid];
      const { idx, p, byIdx } = policy(this.net, game, pid, this.win);
      if (actionsThisTurn >= 250 && !pl.pending.length) return game.legalActions().find((a) => a.kind === AK.END_TURN);
      this.stats.decisions++;
      const dflt = byIdx.get(idx[pick(p, this.rand)]);
      const acts = idx.map((i) => byIdx.get(i));
      if (idx.length < 2 || Math.max(...p) >= this.confident || (this.gate === "critical" && !this.critical(game, pid, acts))) return dflt;
      const order = idx.map((_, j) => j).sort((a, b) => p[b] - p[a]).slice(0, this.k);
      const cands = order.map((j) => acts[j]);
      if (cands.some((a) => a.kind === AK.UNIT && a.arg === UA.EXAMINE)) return dflt;
      const t0 = (typeof performance !== "undefined" ? performance : Date).now();
      // All k x m continuations advance together, one batched network call per step
      // (each still follows only its own policy; the first per candidate is greedy).
      const base = this.knownCopy(game, pid), runs = [];
      cands.forEach((c, j) => {
        if (c.kind === AK.END_TURN) { runs.push({ j, g: null, greedy: true, val: null, end: true }); return; }
        for (let r = 0; r < this.m; r++) {
          const g = cloneGame(base, Math.floor(this.rand() * 4294967296));
          g.apply(c);
          runs.push({ j, g, greedy: r === 0, val: null, n: 0 });
        }
      });
      const endTurnVal = runs.some((u) => u.end) ? policy(this.net, base, pid, this.win).value : 0;
      for (const u of runs) if (u.end) u.val = endTurnVal;
      const outcome = (g) => (g.result.winner === pid ? 1 : -1);
      for (;;) {
        for (const u of runs) if (u.val === null && (u.g.done)) u.val = outcome(u.g);
        const live = runs.filter((u) => u.val === null && u.g.current === pid && u.n < this.maxActions);
        const left = runs.filter((u) => u.val === null && !live.includes(u));
        if (left.length) policies(this.net, left.map((u) => u.g), pid, this.win).forEach((q, i) => { left[i].val = q.value; });
        if (!live.length) break;
        policies(this.net, live.map((u) => u.g), pid, this.win).forEach(({ idx, p, value, byIdx }, i) => {
          const u = live[i];
          const jj = u.greedy ? p.indexOf(Math.max(...p)) : pick(p, this.rand);
          const a = byIdx.get(idx[jj]);
          if (a.kind === AK.END_TURN || (a.kind === AK.UNIT && a.arg === UA.EXAMINE)) { u.val = value; return; }
          u.g.apply(a); u.n++;
        });
      }
      const scores = cands.map((_, j) => { const v = runs.filter((u) => u.j === j).map((u) => u.val); return v.reduce((x, y) => x + y, 0) / v.length; });
      let best = 0;
      for (let j = 1; j < cands.length; j++) if (scores[j] > scores[best]) best = j;
      if (scores[best] < scores[0] + this.margin) best = 0;
      this.stats.searched++;
      this.stats.changed += best !== 0 ? 1 : 0;
      this.stats.ms += (typeof performance !== "undefined" ? performance : Date).now() - t0;
      this.lastScores = cands.map((a, j) => [a, scores[j]]);
      return cands[best];
    }
  }

  const api = { PolicyNet, TfNet, fastNet, netAction, expertAction, sampleMasked, SearchAgent, cloneGame, policies, attachWinHead };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PolyAI = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
