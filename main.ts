import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const STACK_MAX = 256;
const CALLS_MAX = 64;
const MEM_WORDS = 256;
// Instruction execution step budget per VM run
const STEP_LIMIT = 1_000_000;
const OUT_MAX = 65_536;

const TRAP = {
  UNDERFLOW: 2, OVERFLOW: 3, DIV_ZERO: 4, BAD_OP: 5, MEMORY: 6,
  CALL_DEPTH: 7, RET_EMPTY: 8, PC: 9, STEPS: 10, OUTPUT: 11,
};

const OPS: Record<number, [number, number, number]> = {
  0x00: [0, 0, 0], 0x01: [1, 0, 1], 0x02: [4, 0, 1], 0x03: [0, 1, 0], 0x04: [0, 1, 2],
  0x05: [0, 2, 2], 0x06: [0, 2, 3], 0x07: [0, 3, 3], 0x10: [0, 2, 1], 0x11: [0, 2, 1],
  0x12: [0, 2, 1], 0x13: [0, 2, 1], 0x14: [0, 2, 1], 0x15: [0, 2, 1], 0x16: [0, 2, 1],
  0x17: [0, 1, 1], 0x18: [0, 2, 1], 0x19: [0, 2, 1], 0x1a: [0, 2, 1], 0x1b: [0, 1, 1],
  0x1c: [0, 2, 1], 0x1d: [0, 2, 1], 0x1e: [0, 2, 1], 0x20: [0, 2, 1], 0x21: [0, 2, 1],
  0x22: [0, 2, 1], 0x30: [2, 0, 0], 0x31: [2, 1, 0], 0x32: [2, 1, 0], 0x33: [2, 0, 0],
  0x34: [0, 0, 0], 0x40: [0, 1, 1], 0x41: [0, 2, 0], 0x50: [0, 1, 0], 0x51: [0, 1, 0],
  0x52: [0, 0, 1], 0x53: [0, 1, 0],
};

const enc = new TextEncoder();

function evalAlu(op: number, a: number, b: number): number | 'DIV_ZERO' {
  if (b === 0 && op >= 0x13 && op <= 0x16) return 'DIV_ZERO';
  const ua = a >>> 0, ub = b >>> 0, s = b & 31;
  switch (op) {
    case 0x10: return (a + b) | 0;
    case 0x11: return (a - b) | 0;
    case 0x12: return Math.imul(a, b);
    case 0x13: return a === -2147483648 && b === -1 ? a : (a / b) | 0;
    case 0x14: return b === -1 ? 0 : (a % b) | 0;
    case 0x15: return Math.floor(ua / ub) | 0;
    case 0x16: return (ua % ub) | 0;
    case 0x18: return a & b;
    case 0x19: return a | b;
    case 0x1a: return a ^ b;
    case 0x1c: return a << s;
    case 0x1d: return a >> s;
    case 0x1e: return (ua >>> s) | 0;
    case 0x20: return a === b ? 1 : 0;
    case 0x21: return a < b ? 1 : 0;
    case 0x22: return ua < ub ? 1 : 0;
    default: return 0;
  }
}

function evalBranch(op: number, prog: Uint8Array, next: number, cond: number): { take: boolean; target: number } {
  const rel = ((prog[next - 2] | (prog[next - 1] << 8)) << 16) >> 16;
  const take = op === 0x30 || (op === 0x31 ? cond === 0 : cond !== 0);
  return { take, target: next + rel };
}

function accessMem(M: Int32Array, addr: number, val?: number): number | 'ERR' {
  if (addr < 0 || addr >= MEM_WORDS) return 'ERR';
  if (val !== undefined) { M[addr] = val; return 0; }
  return M[addr];
}

function writeOut(val: number, isChar: boolean, out: Uint8Array, olen: number): number | 'ERR' {
  if (isChar) {
    if (olen + 1 > OUT_MAX) return 'ERR';
    out[olen] = val & 255;
    return olen + 1;
  }
  const b = enc.encode(String(val));
  if (olen + b.length > OUT_MAX) return 'ERR';
  out.set(b, olen);
  return olen + b.length;
}

function run(prog: Uint8Array, inp: Uint8Array): { stdout: Uint8Array; exit: number } {
  const n = prog.length;
  const D = new Int32Array(STACK_MAX);
  let sp = 0;
  const C = new Int32Array(CALLS_MAX);
  let csp = 0;
  const M = new Int32Array(MEM_WORDS);
  const out = new Uint8Array(OUT_MAX);
  let olen = 0;
  let inPos = 0;
  let pc = 0;
  let steps = 0;

  const done = (exit: number) => ({ stdout: out.slice(0, olen), exit });
  const trap = (code: number) => done(code);

  for (;;) {
    if (steps === STEP_LIMIT) return trap(TRAP.STEPS);
    if (pc < 0 || pc >= n) return trap(TRAP.PC);

    const op = prog[pc];
    const info = OPS[op];
    if (!info) return trap(TRAP.BAD_OP);

    const [imm, pops, pushes] = info;
    if (pc + imm >= n) return trap(TRAP.PC);

    steps++;
    const next = pc + 1 + imm;

    if (sp < pops) return trap(TRAP.UNDERFLOW);
    if (sp - pops + pushes > STACK_MAX) return trap(TRAP.OVERFLOW);

    pc = next;

    switch (op) {
      case 0x00: return done(0);
      case 0x01: D[sp++] = (prog[next - 1] << 24) >> 24; break;
      case 0x02: D[sp++] = prog[next - 4] | (prog[next - 3] << 8) | (prog[next - 2] << 16) | (prog[next - 1] << 24); break;
      case 0x03: sp--; break;
      case 0x04: D[sp] = D[sp - 1]; sp++; break;
      case 0x05: { const t = D[sp - 1]; D[sp - 1] = D[sp - 2]; D[sp - 2] = t; break; }
      case 0x06: D[sp] = D[sp - 2]; sp++; break;
      case 0x07: { const t = D[sp - 3]; D[sp - 3] = D[sp - 2]; D[sp - 2] = D[sp - 1]; D[sp - 1] = t; break; }
      case 0x17: D[sp - 1] = -D[sp - 1] | 0; break;
      case 0x1b: D[sp - 1] = ~D[sp - 1]; break;
      case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16:
      case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d: case 0x1e:
      case 0x20: case 0x21: case 0x22: {
        const res = evalAlu(op, D[sp - 2], D[sp - 1]);
        if (res === 'DIV_ZERO') return trap(TRAP.DIV_ZERO);
        sp--;
        D[sp - 1] = res;
        break;
      }
      case 0x30: case 0x31: case 0x32: {
        const { take, target } = evalBranch(op, prog, next, op === 0x30 ? 0 : D[--sp]);
        if (take) pc = target;
        break;
      }
      case 0x33:
        if (csp === CALLS_MAX) return trap(TRAP.CALL_DEPTH);
        C[csp++] = next;
        pc = prog[next - 2] | (prog[next - 1] << 8);
        break;
      case 0x34:
        if (csp === 0) return trap(TRAP.RET_EMPTY);
        pc = C[--csp];
        break;
      case 0x40: {
        const v = accessMem(M, D[sp - 1]);
        if (v === 'ERR') return trap(TRAP.MEMORY);
        D[sp - 1] = v;
        break;
      }
      case 0x41: {
        if (accessMem(M, D[sp - 1], D[sp - 2]) === 'ERR') return trap(TRAP.MEMORY);
        sp -= 2;
        break;
      }
      case 0x50: case 0x51: {
        const res = writeOut(D[--sp], op === 0x51, out, olen);
        if (res === 'ERR') return trap(TRAP.OUTPUT);
        olen = res;
        break;
      }
      case 0x52:
        D[sp++] = inPos < inp.length ? inp[inPos++] : -1;
        break;
      case 0x53:
        return done(D[--sp] & 255);
    }
  }
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const id of readdirSync(I).sort()) {
  const p = new Uint8Array(readFileSync(join(I, id, 'program.bin')));
  const inp = new Uint8Array(readFileSync(join(I, id, 'input.bin')));
  const r = run(p, inp);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), r.stdout);
  writeFileSync(join(O, id, 'exit.txt'), `${r.exit}\n`);
}
