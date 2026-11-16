# bytecode-vm — Compiler + Stack Virtual Machine

Source → bytecode → execution, with a disassembler. A small C-like language is
compiled in a single pass (jump backpatching, call-site linking) to typed
opcodes and executed by a stack VM with an instruction budget — and
cross-checked against a tree-walking **reference evaluator** so the compiler is
proven against an independent implementation.

## Why it's special

- **Two engines, one language.** Every test runs the same program through the
  compiled VM *and* a direct AST interpreter and asserts identical output —
  this is how the jump logic, short-circuit codegen, and implicit returns stay
  honest.
- **Real compiler machinery.** Typed opcodes (`CONST_I`/`CONST_F`/`CONST_S`/
  `CONST_B`, jumps, calls, returns), forward jumps patched at close, backward
  loop edges, forward function calls linked after all bodies compile, arity
  checked at compile time with line numbers.
- **Budget, not hangs.** The VM counts instructions; `while (true) { }` fails
  fast with `VMError: instruction budget exceeded` instead of freezing.
- **Inspectable.** `dis` prints annotated bytecode; golden tests pin the exact
  output. Stack depth is tracked (`peak_depth`) and underflow is structurally
  impossible: every operand-count guard runs *before* any pop, and a
  hand-corrupted program is rejected with the stack untouched.

## Architecture

```
bytecode_vm.py
├── lexer      line/column tokens, numbers, strings, // comments, keywords
├── parser     recursive descent, precedence climbing (|| && == rel add mul unary)
├── AST        dataclass nodes, every node carries its source line
├── Compiler   single pass → Op(code, arg, line); backpatch + pending-call linking
├── VM         stack machine, call frames, budget, peak-depth tracking
├── reference  tree-walking evaluator used for differential testing
├── disassembler  stable annotated listing (golden-tested)
└── CLI        run | dis, --budget N
```

## Language

```c
// comment
let x = 1 + 2 * 3;      // precedence, ints and floats
x = x - 1;              // assignment
print x;                // output

if (x > 5) { print "big"; }
elif (x == 5) { print "five"; }
else { print "small"; }

while (x < 10) { x = x + 1; }

fn add(a, b) { return a + b; }
print add(3, 4);
```

Semantics (documented): `+` concatenates two strings; `/` is true division;
`%` follows Python remainder rules; comparisons work on numbers or on two
strings; `==`/`!=` compare any values; `&&`/`||` short-circuit and return
their operand (not just a bool); truthiness is Python-like (`false`, `0`, `""`
and end-of-program `none` are falsy); a function that falls off the end
returns `false`. Functions are declared at top level only (no closures);
locals are function-scoped, `let` redeclaration reuses the slot.

## Runnable example

```console
> python bytecode_vm.py run example.bvm
0
1
1
2
3
5
8
13
21
34
55
```

```console
> python bytecode_vm.py dis example.bvm
0000  CONST_I       0           ; push int literal
0001  STORE_GLOBAL  i           ; store global
0002  POP                       ; discard top
0003  LOAD_GLOBAL   i           ; load global
0004  CONST_I       10          ; push int literal
0005  LE                        ; less or equal
0006  JMP_IF_FALSE  0016        ; pop; jump if false
...
```

## Tests

```console
> python -m unittest discover -s tests -v
```

13 tests: an 8-program corpus checked against the reference evaluator *and*
hand-computed output; jump-target validation (forward + backpatched back
edges); compile/parse/lex errors with line and column; arity errors on both
engines; instruction budget; golden disassembly; stack peak/underflow guards;
local/global scoping; CLI `run`/`dis`/`--budget`.
