import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// VM architectural limits
const STACK_MAX = 256;
const CALLS_MAX = 64;
const MEM_WORDS = 256;
const STEP_LIMIT = 1_000_000;
const OUT_MAX = 65_536;

// Trap exit codes defined in TinyStack VM ISA specification
const TRAP = {
  UNDERFLOW: 2,
  OVERFLOW: 3,
  DIV_ZERO: 4,
  BAD_OP: 5,
  MEMORY: 6,
  CALL_DEPTH: 7,
  RET_EMPTY: 8,
  PC: 9,
  STEPS: 10,
  OUTPUT: 11,
};

// Opcode metadata: [immediate_bytes, stack_pops, stack_pushes]
const OPS: Record<number, [number, number, number]> = {
  0x00: [0, 0, 0], // HALT
  0x01: [1, 0, 1], // PUSH8
  0x02: [4, 0, 1], // PUSH32
  0x03: [0, 1, 0], // POP
  0x04: [0, 1, 2], // DUP
  0x05: [0, 2, 2], // SWAP
  0x06: [0, 2, 3], // OVER
  0x07: [0, 3, 3], // ROT
  0x10: [0, 2, 1], // ADD
  0x11: [0, 2, 1], // SUB
  0x12: [0, 2, 1], // MUL
  0x13: [0, 2, 1], // DIV_S
  0x14: [0, 2, 1], // MOD_S
  0x15: [0, 2, 1], // DIV_U
  0x16: [0, 2, 1], // MOD_U
  0x17: [0, 1, 1], // NEG
  0x18: [0, 2, 1], // AND
  0x19: [0, 2, 1], // OR
  0x1a: [0, 2, 1], // XOR
  0x1b: [0, 1, 1], // NOT
  0x1c: [0, 2, 1], // SHL
  0x1d: [0, 2, 1], // SHR_S
  0x1e: [0, 2, 1], // SHR_U
  0x20: [0, 2, 1], // EQ
  0x21: [0, 2, 1], // LT_S
  0x22: [0, 2, 1], // LT_U
  0x30: [2, 0, 0], // JMP
  0x31: [2, 1, 0], // JZ
  0x32: [2, 1, 0], // JNZ
  0x33: [2, 0, 0], // CALL
  0x34: [0, 0, 0], // RET
  0x40: [0, 1, 1], // LOAD
  0x41: [0, 2, 0], // STORE
  0x50: [0, 1, 0], // PRINT
  0x51: [0, 1, 0], // PUTC
  0x52: [0, 0, 1], // IN
  0x53: [0, 1, 0], // EXIT
};

const enc = new TextEncoder();

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

    // Fast-path opcode dispatch and stack bounds validation
    if (sp < pops) return trap(TRAP.UNDERFLOW);
    if (sp - pops + pushes > STACK_MAX) return trap(TRAP.OVERFLOW);

    pc = next;

    // Single-pass switch dispatch: handles literal push, ALU, memory, jumps, calls, and I/O
    switch (op) {
      case 0x00: return done(0); // HALT
      case 0x01: // PUSH8: sign-extended 8-bit immediate
        D[sp++] = (prog[next - 1] << 24) >> 24;
        break;
      case 0x02: // PUSH32: 32-bit little-endian immediate
        D[sp++] = prog[next - 4] | (prog[next - 3] << 8) | (prog[next - 2] << 16) | (prog[next - 1] << 24);
        break;
      case 0x03: // POP
        sp--;
        break;
      case 0x04: // DUP
        D[sp] = D[sp - 1];
        sp++;
        break;
      case 0x05: { // SWAP
        const t = D[sp - 1];
        D[sp - 1] = D[sp - 2];
        D[sp - 2] = t;
        break;
      }
      case 0x06: // OVER
        D[sp] = D[sp - 2];
        sp++;
        break;
      case 0x07: { // ROT: ( a b c -- b c a )
        const a = D[sp - 3];
        D[sp - 3] = D[sp - 2];
        D[sp - 2] = D[sp - 1];
        D[sp - 1] = a;
        break;
      }
      case 0x17: // NEG
        D[sp - 1] = -D[sp - 1] | 0;
        break;
      case 0x1b: // NOT
        D[sp - 1] = ~D[sp - 1];
        break;
      case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16:
      case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d: case 0x1e:
      case 0x20: case 0x21: case 0x22: {
        const b = D[sp - 1], a = D[sp - 2];
        let r = 0;
        if (op === 0x10) r = (a + b) | 0;
        else if (op === 0x11) r = (a - b) | 0;
        else if (op === 0x12) r = Math.imul(a, b);
        else if (op === 0x13) {
          if (b === 0) return trap(TRAP.DIV_ZERO);
          r = a === -2147483648 && b === -1 ? a : (a / b) | 0;
        }
        else if (op === 0x14) {
          if (b === 0) return trap(TRAP.DIV_ZERO);
          r = b === -1 ? 0 : (a % b) | 0;
        }
        else if (op === 0x15) {
          if (b === 0) return trap(TRAP.DIV_ZERO);
          r = Math.floor((a >>> 0) / (b >>> 0)) | 0;
        }
        else if (op === 0x16) {
          if (b === 0) return trap(TRAP.DIV_ZERO);
          r = ((a >>> 0) % (b >>> 0)) | 0;
        }
        else if (op === 0x18) r = a & b;
        else if (op === 0x19) r = a | b;
        else if (op === 0x1a) r = a ^ b;
        else if (op === 0x1c) r = a << (b & 31);
        else if (op === 0x1d) r = a >> (b & 31);
        else if (op === 0x1e) r = (a >>> (b & 31)) | 0;
        else if (op === 0x20) r = a === b ? 1 : 0;
        else if (op === 0x21) r = a < b ? 1 : 0;
        else if (op === 0x22) r = (a >>> 0) < (b >>> 0) ? 1 : 0;
        sp -= 2;
        D[sp++] = r;
        break;
      }
      case 0x30: case 0x31: case 0x32: { // Control flow jumps
        const rel = ((prog[next - 2] | (prog[next - 1] << 8)) << 16) >> 16;
        let take = true;
        if (op !== 0x30) {
          const v = D[--sp];
          take = op === 0x31 ? v === 0 : v !== 0;
        }
        if (take) pc = next + rel;
        break;
      }
      case 0x33: // CALL
        if (csp === CALLS_MAX) return trap(TRAP.CALL_DEPTH);
        C[csp++] = next;
        pc = prog[next - 2] | (prog[next - 1] << 8);
        break;
      case 0x34: // RET
        if (csp === 0) return trap(TRAP.RET_EMPTY);
        pc = C[--csp];
        break;
      case 0x40: { // LOAD
        const addr = D[sp - 1];
        if (addr < 0 || addr >= MEM_WORDS) return trap(TRAP.MEMORY);
        D[sp - 1] = M[addr];
        break;
      }
      case 0x41: { // STORE
        const addr = D[sp - 1], v = D[sp - 2];
        if (addr < 0 || addr >= MEM_WORDS) return trap(TRAP.MEMORY);
        sp -= 2;
        M[addr] = v;
        break;
      }
      case 0x50: { // PRINT (signed decimal ASCII representation)
        const bytes = enc.encode(String(D[sp - 1]));
        if (olen + bytes.length > OUT_MAX) return trap(TRAP.OUTPUT);
        sp--;
        out.set(bytes, olen);
        olen += bytes.length;
        break;
      }
      case 0x51: // PUTC
        if (olen + 1 > OUT_MAX) return trap(TRAP.OUTPUT);
        out[olen++] = D[--sp] & 255;
        break;
      case 0x52: // IN
        D[sp++] = inPos < inp.length ? inp[inPos++] : -1;
        break;
      case 0x53: // EXIT
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
