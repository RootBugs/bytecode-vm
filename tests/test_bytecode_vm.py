"""Tests for bytecode-vm: compiler, stack VM, disassembler, reference evaluator."""

import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bytecode_vm as bvm
from bytecode_vm import (
    CompileError,
    EvalError,
    LexError,
    Op,
    OpCode,
    ParseError,
    Program,
    VM,
    VMError,
    compile_source,
    disassemble_source,
    parse_source,
    reference_run_source,
    run_source,
)

# Program corpus: every program runs on both engines and must produce the
# hand-computed output below.
CORPUS = [
    ("print 1 + 2 * 3 - 4 / 2;", "5.0\n"),
    (
        "let total = 0;\nlet i = 1;\n"
        "while (i <= 10) { total = total + i; i = i + 1; }\n"
        "print total;",
        "55\n",
    ),
    (
        'let x = 7;\n'
        'if (x > 10) { print "big"; } elif (x > 5) { print "mid"; }'
        ' else { print "small"; }',
        "mid\n",
    ),
    (
        "fn add(a, b) { return a + b; }\n"
        "fn fib(n) { if (n < 2) { return n; } return fib(n - 1) + fib(n - 2); }\n"
        "print add(2, 3);\nprint fib(10);",
        "5\n55\n",
    ),
    (
        'let s = "he" + "llo";\nprint s;\nprint s == "hello";\nprint !false && true;',
        "hello\ntrue\ntrue\n",
    ),
    (
        'let rows = "";\nlet i = 0;\n'
        "while (i < 3) {\n"
        "  let j = 0;\n"
        '  while (j < 3) { rows = rows + "*"; j = j + 1; }\n'
        '  rows = rows + "|";\n'
        "  i = i + 1;\n"
        "}\nprint rows;",
        "***|***|***|\n",
    ),
    ("print 7 % 3;\nprint 7 / 2;\nprint 7 / 7;", "1\n3.5\n1.0\n"),
    ("let a = 3;\nlet b = 4;\nprint a * b >= 12 && a < b;", "true\n"),
]


class CorpusTests(unittest.TestCase):
    def test_corpus_matches_reference_and_expected(self):
        for src, expected in CORPUS:
            with self.subTest(src=src.splitlines()[0]):
                vm_out = run_source(src)
                ref_out = reference_run_source(src)
                self.assertEqual(vm_out, expected)
                self.assertEqual(ref_out, expected)
                self.assertEqual(vm_out, ref_out)

    def test_vm_and_reference_agree_on_edge_values(self):
        src = (
            'print 1 == 1.0;\nprint "a" < "b";\nprint 0 || 7;\n'
            "print 0 && 9;\nprint -(-3);\nprint !(1 < 0);"
        )
        expected = "true\ntrue\n7\nfalse\n3\ntrue\n"
        self.assertEqual(run_source(src), expected)
        self.assertEqual(reference_run_source(src), expected)


class CompilerTests(unittest.TestCase):
    def test_jump_targets_valid_and_backpatched(self):
        prog = compile_source(
            "let i = 0;\nwhile (i < 3) {\n"
            "  if (i == 1) { print 1; } elif (i == 2) { print 2; }"
            " else { print 0; }\n"
            "  i = i + 1;\n}"
        )
        back_edges = 0
        forward = 0
        for ip, op in enumerate(prog.code):
            if op.code in (OpCode.JMP, OpCode.JMP_IF_FALSE, OpCode.JMP_IF_TRUE):
                self.assertIsInstance(op.arg, int)
                self.assertGreaterEqual(op.arg, 0)
                self.assertLess(op.arg, len(prog.code))
                if op.arg <= ip:
                    back_edges += 1
                else:
                    forward += 1
        self.assertGreater(back_edges, 0)  # while loop back edge was backpatched
        self.assertGreater(forward, 0)  # conditional exits were backpatched

    def test_function_arity_errors(self):
        src = "fn add(a, b) { return a + b; }\nprint add(1);"
        with self.assertRaises(CompileError) as cm:
            compile_source(src)
        self.assertIn("function 'add' expects 2 arguments, got 1", cm.exception.message)
        self.assertEqual(cm.exception.line, 2)
        with self.assertRaises(EvalError) as cm:
            reference_run_source(src)
        self.assertIn("function 'add' expects 2 arguments, got 1", cm.exception.message)
        # Forward references are linked after all function bodies compile.
        fwd = "print later(1);\nfn later(x) { return x; }"
        self.assertEqual(run_source(fwd), "1\n")

    def test_compile_time_error_messages(self):
        with self.assertRaises(CompileError) as cm:
            compile_source("y = 3;")
        self.assertIn("undefined variable 'y'", cm.exception.message)
        self.assertEqual(cm.exception.line, 1)
        with self.assertRaises(CompileError) as cm:
            compile_source("return 1;")
        self.assertIn("'return' outside of a function", cm.exception.message)
        with self.assertRaises(ParseError) as cm:
            compile_source("if (true) { fn f() { return 1; } }")
        self.assertIn("only allowed at top level", cm.exception.message)
        with self.assertRaises(ParseError) as cm:
            compile_source("fn f(a, a) { return a; }")
        self.assertIn("duplicate parameter", cm.exception.message)
        with self.assertRaises(CompileError) as cm:
            compile_source("fn f() { return 1; }\nfn f() { return 2; }")
        self.assertIn("duplicate definition of 'f'", cm.exception.message)

    def test_parse_and_lex_errors_carry_position(self):
        with self.assertRaises(ParseError) as cm:
            parse_source("let x = ;")
        self.assertIn("expected expression", cm.exception.message)
        self.assertEqual(cm.exception.line, 1)
        with self.assertRaises(ParseError) as cm:
            parse_source("print 1 +\n2 *;")
        self.assertEqual(cm.exception.line, 2)
        with self.assertRaises(ParseError) as cm:
            parse_source("while (true) { print 1;")
        self.assertIn("unterminated block", cm.exception.message)
        with self.assertRaises(LexError) as cm:
            parse_source("let x = 1;\nprint x @;")
        self.assertIn("unexpected character '@'", cm.exception.message)
        self.assertEqual(cm.exception.line, 2)
        self.assertEqual(cm.exception.col, 9)


class VmTests(unittest.TestCase):
    def test_instruction_budget_stops_infinite_loop(self):
        prog = compile_source("while (true) { }")
        vm = VM(prog, stdout=io.StringIO(), budget=500)
        with self.assertRaises(VMError) as cm:
            vm.run()
        self.assertIn("instruction budget exceeded", cm.exception.message)
        self.assertIsNotNone(cm.exception.line)

    def test_runtime_errors_report_line_and_message(self):
        with self.assertRaises(VMError) as cm:
            run_source("print 1;\nprint 1 / 0;")
        self.assertIn("division by zero", cm.exception.message)
        self.assertEqual(cm.exception.line, 2)
        with self.assertRaises(VMError) as cm:
            run_source('print 1 + "a";')
        self.assertIn("invalid operands for '+'", cm.exception.message)
        with self.assertRaises(EvalError) as cm:
            reference_run_source("print 1;\nprint 1 / 0;")
        self.assertIn("division by zero", cm.exception.message)
        self.assertEqual(cm.exception.line, 2)
        with self.assertRaises(CompileError) as cm:
            run_source("print missing;")
        self.assertIn("undefined variable 'missing'", cm.exception.message)

    def test_stack_depth_tracked_and_underflow_impossible(self):
        prog = compile_source("print 1 + 2 * 3;")
        vm = VM(prog, stdout=io.StringIO())
        vm.run()
        self.assertEqual(vm.peak_depth, 3)  # 1, then 2, then 3 pushed
        self.assertEqual(vm.stack, [])
        self.assertEqual(vm.frames, [])
        self.assertGreater(vm.steps, 0)
        # A hand-corrupted program cannot pop: the guard fires first and the
        # stack is left untouched.
        bad = Program(code=[Op(OpCode.SUB, None, 7)], functions={})
        vm = VM(bad, stdout=io.StringIO())
        with self.assertRaises(VMError) as cm:
            vm.run()
        self.assertIn("stack underflow", cm.exception.message)
        self.assertEqual(cm.exception.line, 7)
        self.assertEqual(vm.stack, [])

    def test_scoping_locals_and_globals(self):
        src = (
            "let g = 10;\n"
            "fn bump(n) { g = g + n; return g; }\n"
            "print bump(5);\nprint g;\n"
            "let x = 1;\n"
            "fn f() { let x = 2; return x; }\n"
            "print f();\nprint x;"
        )
        expected = "15\n15\n2\n1\n"
        self.assertEqual(run_source(src), expected)
        self.assertEqual(reference_run_source(src), expected)


class DisassemblerTests(unittest.TestCase):
    def test_disassembler_golden_output(self):
        expected = (
            "0000  CONST_I       1           ; push int literal\n"
            "0001  CONST_I       2           ; push int literal\n"
            "0002  ADD                       ; add\n"
            "0003  PRINT                     ; print\n"
            "0004  HALT                      ; stop\n"
        )
        self.assertEqual(disassemble_source("print 1 + 2;"), expected)

    def test_disassembler_annotates_control_flow_and_calls(self):
        out = disassemble_source(
            "fn twice(n) { return n * 2; }\n"
            "if (true) { print twice(4); } else { print 0; }"
        )
        self.assertIn("JMP_IF_FALSE", out)
        self.assertIn("twice/1@", out)
        self.assertIn("CALL", out)
        self.assertIn("RET", out)
        self.assertIn("jump if false", out)
        for line in out.splitlines():
            self.assertRegex(line, r"^\d{4}  [A-Z_]+ +.*; .+$")


class CliTests(unittest.TestCase):
    def test_run_and_dis_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prog.bvm")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("print 40 + 2;\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = bvm.main(["run", path])
            self.assertEqual(code, 0)
            self.assertEqual(buf.getvalue(), "42\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = bvm.main(["dis", path])
            self.assertEqual(code, 0)
            self.assertIn("CONST_I", buf.getvalue())
            self.assertIn("HALT", buf.getvalue())
            # Usage for unknown mode.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = bvm.main(["frobnicate", path])
            self.assertEqual(code, 2)
            self.assertIn("usage:", buf.getvalue())
            # Budget flag honoured.
            loop_path = os.path.join(tmp, "loop.bvm")
            with open(loop_path, "w", encoding="utf-8") as fh:
                fh.write("while (true) { }\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = bvm.main(["run", loop_path, "--budget", "200"])
            self.assertEqual(code, 1)
            self.assertIn("instruction budget exceeded", err.getvalue())


if __name__ == "__main__":
    unittest.main()
