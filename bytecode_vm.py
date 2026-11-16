"""bytecode-vm — source → bytecode → execution, with a disassembler.

A small C-like language compiled to typed opcodes and executed by a stack VM
with an instruction budget, plus a tree-walking reference evaluator used to
cross-check compiled output, and a disassembler for inspection.

Language subset: arithmetic with precedence, variables and assignment,
``if/elif/else``, ``while``, functions with locals, ``print`` and ``return``.
Comments run from ``//`` to end of line.  Statements end with ``;``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "LexError",
    "ParseError",
    "CompileError",
    "VMError",
    "EvalError",
    "Program",
    "compile_source",
    "run_source",
    "reference_run_source",
    "disassemble",
    "disassemble_source",
    "main",
]


# ---------------------------------------------------------------- errors ---


class SourceError(Exception):
    """Base class for errors that carry a 1-based source line."""

    def __init__(self, message: str, line: Optional[int] = None, col: Optional[int] = None):
        self.message = message
        self.line = line
        self.col = col
        location = ""
        if line is not None:
            location = f" [line {line}" + (f":{col}]" if col is not None else "]")
        super().__init__(message + location)


class LexError(SourceError):
    """Illegal character or malformed token."""


class ParseError(SourceError):
    """Syntax error."""


class CompileError(SourceError):
    """Semantic error found while compiling (undefined variable, arity …)."""


class VMError(SourceError):
    """Runtime error inside the virtual machine."""


class EvalError(SourceError):
    """Runtime error inside the reference evaluator."""


# ---------------------------------------------------------------- lexer ----

KEYWORDS = {"let", "if", "elif", "else", "while", "fn", "return", "print", "true", "false"}

_SINGLE = set("()+-*/%{};,=<>!")
_TWO = {"==", "!=", "<=", ">=", "&&", "||"}


@dataclass
class Token:
    kind: str  # NUM, STR, IDENT, KW, OP, EOF
    value: Any
    line: int
    col: int


def tokenize(src: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line = 1
    line_start = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\n":
            i += 1
            line += 1
            line_start = i
            continue
        if c in " \t\r":
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        col = i - line_start + 1
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            start = i
            seen_dot = False
            seen_exp = False
            while i < n and (src[i].isdigit() or src[i] in ".eE" or (src[i] in "+-" and seen_exp and src[i - 1] in "eE")):
                if src[i] == ".":
                    if seen_dot or seen_exp:
                        break
                    seen_dot = True
                elif src[i] in "eE":
                    if seen_exp:
                        break
                    seen_exp = True
                i += 1
            text = src[start:i]
            try:
                value: Any = int(text) if not seen_dot and not seen_exp else float(text)
            except ValueError:
                raise LexError(f"invalid number literal '{text}'", line, col) from None
            tokens.append(Token("NUM", value, line, col))
            continue
        if c == '"':
            i += 1
            buf: List[str] = []
            closed = False
            while i < n:
                ch = src[i]
                if ch == "\\":
                    if i + 1 >= n:
                        raise LexError("unterminated string literal", line, col)
                    esc = src[i + 1]
                    mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "0": "\0"}
                    if esc not in mapping:
                        raise LexError(f"invalid escape '\\{esc}'", line, col)
                    buf.append(mapping[esc])
                    i += 2
                elif ch == '"':
                    i += 1
                    closed = True
                    break
                elif ch == "\n":
                    raise LexError("unterminated string literal", line, col)
                else:
                    buf.append(ch)
                    i += 1
            if not closed:
                raise LexError("unterminated string literal", line, col)
            tokens.append(Token("STR", "".join(buf), line, col))
            continue
        if c.isalpha() or c == "_":
            start = i
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            word = src[start:i]
            kind = "KW" if word in KEYWORDS else "IDENT"
            tokens.append(Token(kind, word, line, col))
            continue
        two = src[i : i + 2]
        if two in _TWO:
            tokens.append(Token("OP", two, line, col))
            i += 2
            continue
        if c in _SINGLE:
            tokens.append(Token("OP", c, line, col))
            i += 1
            continue
        raise LexError(f"unexpected character '{c}'", line, col)
    tokens.append(Token("EOF", None, line, 1))
    return tokens


# ------------------------------------------------------------- AST nodes ---


@dataclass
class Num:
    value: Any
    line: int


@dataclass
class Str:
    value: str
    line: int


@dataclass
class Bool:
    value: bool
    line: int


@dataclass
class Var:
    name: str
    line: int


@dataclass
class Unary:
    op: str
    operand: Any
    line: int


@dataclass
class Binary:
    op: str
    left: Any
    right: Any
    line: int


@dataclass
class Call:
    name: str
    args: List[Any]
    line: int


@dataclass
class Let:
    name: str
    value: Any
    line: int


@dataclass
class Assign:
    name: str
    value: Any
    line: int


@dataclass
class Print:
    expr: Any
    line: int


@dataclass
class If:
    branches: List[Tuple[Any, "Block"]]  # (condition, block)
    else_block: Optional["Block"]
    line: int


@dataclass
class While:
    cond: Any
    body: "Block"
    line: int


@dataclass
class Function:
    name: str
    params: List[str]
    body: "Block"
    line: int


@dataclass
class Return:
    expr: Any
    line: int


@dataclass
class Block:
    stmts: List[Any]
    line: int


@dataclass
class ExprStmt:
    expr: Any
    line: int


# --------------------------------------------------------------- parser ----


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def cur(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def check_op(self, *ops: str) -> bool:
        tok = self.cur
        return tok.kind == "OP" and tok.value in ops

    def match_op(self, *ops: str) -> Optional[Token]:
        if self.check_op(*ops):
            return self.advance()
        return None

    def match_kw(self, word: str) -> Optional[Token]:
        tok = self.cur
        if tok.kind == "KW" and tok.value == word:
            return self.advance()
        return None

    def expect_op(self, op: str, what: str) -> Token:
        tok = self.cur
        if tok.kind == "OP" and tok.value == op:
            return self.advance()
        raise ParseError(f"expected '{op}' {what}, found {self._describe(tok)}", tok.line, tok.col)

    def expect_ident(self, what: str) -> Token:
        tok = self.cur
        if tok.kind == "IDENT":
            return self.advance()
        raise ParseError(f"expected identifier {what}, found {self._describe(tok)}", tok.line, tok.col)

    @staticmethod
    def _describe(tok: Token) -> str:
        if tok.kind == "EOF":
            return "end of input"
        if tok.kind == "STR":
            return f'string "{tok.value}"'
        return f"'{tok.value}'"

    # ---- statements ----
    def parse_program(self, src: str) -> List[Any]:
        stmts: List[Any] = []
        while self.cur.kind != "EOF":
            stmts.append(self.parse_statement(top_level=True))
        return stmts

    def parse_statement(self, top_level: bool) -> Any:
        tok = self.cur
        if tok.kind == "KW" and tok.value == "let":
            self.advance()
            name = self.expect_ident("after 'let'")
            self.expect_op("=", "in 'let' declaration")
            value = self.parse_expression()
            self.expect_op(";", "to end 'let' statement")
            return Let(name.value, value, tok.line)
        if tok.kind == "KW" and tok.value == "print":
            self.advance()
            expr = self.parse_expression()
            self.expect_op(";", "to end 'print' statement")
            return Print(expr, tok.line)
        if tok.kind == "KW" and tok.value == "if":
            return self.parse_if()
        if tok.kind == "KW" and tok.value == "while":
            self.advance()
            self.expect_op("(", "after 'while'")
            cond = self.parse_expression()
            self.expect_op(")", "to close 'while' condition")
            body = self.parse_block()
            return While(cond, body, tok.line)
        if tok.kind == "KW" and tok.value == "fn":
            if not top_level:
                raise ParseError("function declarations are only allowed at top level", tok.line, tok.col)
            return self.parse_function()
        if tok.kind == "KW" and tok.value == "return":
            self.advance()
            expr = self.parse_expression()
            self.expect_op(";", "to end 'return' statement")
            return Return(expr, tok.line)
        if tok.kind == "OP" and tok.value == "{":
            return self.parse_block()
        if tok.kind == "IDENT" and self.tokens[self.pos + 1].kind == "OP" and self.tokens[self.pos + 1].value == "=":
            name = self.advance()
            self.advance()  # '='
            value = self.parse_expression()
            self.expect_op(";", "to end assignment")
            return Assign(name.value, value, tok.line)
        expr = self.parse_expression()
        self.expect_op(";", "to end statement")
        return ExprStmt(expr, tok.line)

    def parse_block(self) -> Block:
        open_tok = self.expect_op("{", "to start a block")
        stmts: List[Any] = []
        while not (self.cur.kind == "OP" and self.cur.value == "}"):
            if self.cur.kind == "EOF":
                raise ParseError("unterminated block: expected '}'", open_tok.line, open_tok.col)
            stmts.append(self.parse_statement(top_level=False))
        self.expect_op("}", "to close the block")
        return Block(stmts, open_tok.line)

    def parse_if(self) -> If:
        tok = self.advance()  # 'if'
        branches: List[Tuple[Any, Block]] = []
        self.expect_op("(", "after 'if'")
        cond = self.parse_expression()
        self.expect_op(")", "to close 'if' condition")
        branches.append((cond, self.parse_block()))
        while self.match_kw("elif"):
            self.expect_op("(", "after 'elif'")
            elif_cond = self.parse_expression()
            self.expect_op(")", "to close 'elif' condition")
            branches.append((elif_cond, self.parse_block()))
        else_block = None
        if self.match_kw("else"):
            else_block = self.parse_block()
        return If(branches, else_block, tok.line)

    def parse_function(self) -> Function:
        tok = self.advance()  # 'fn'
        name = self.expect_ident("after 'fn'")
        self.expect_op("(", "after function name")
        params: List[str] = []
        if not (self.cur.kind == "OP" and self.cur.value == ")"):
            while True:
                param = self.expect_ident("in parameter list")
                params.append(param.value)
                if not self.match_op(","):
                    break
        self.expect_op(")", "to close parameter list")
        body = self.parse_block()
        seen: set = set()
        for p in params:
            if p in seen:
                raise ParseError(f"duplicate parameter '{p}'", tok.line, tok.col)
            seen.add(p)
        return Function(name.value, params, body, tok.line)

    # ---- expressions ----
    def parse_expression(self) -> Any:
        return self.parse_or()

    def parse_or(self) -> Any:
        left = self.parse_and()
        while self.check_op("||"):
            op = self.advance()
            right = self.parse_and()
            left = Binary("||", left, right, op.line)
        return left

    def parse_and(self) -> Any:
        left = self.parse_equality()
        while self.check_op("&&"):
            op = self.advance()
            right = self.parse_equality()
            left = Binary("&&", left, right, op.line)
        return left

    def parse_equality(self) -> Any:
        left = self.parse_comparison()
        while self.check_op("==", "!="):
            op = self.advance()
            right = self.parse_comparison()
            left = Binary(op.value, left, right, op.line)
        return left

    def parse_comparison(self) -> Any:
        left = self.parse_additive()
        while self.check_op("<", "<=", ">", ">="):
            op = self.advance()
            right = self.parse_additive()
            left = Binary(op.value, left, right, op.line)
        return left

    def parse_additive(self) -> Any:
        left = self.parse_multiplicative()
        while self.check_op("+", "-"):
            op = self.advance()
            right = self.parse_multiplicative()
            left = Binary(op.value, left, right, op.line)
        return left

    def parse_multiplicative(self) -> Any:
        left = self.parse_unary()
        while self.check_op("*", "/", "%"):
            op = self.advance()
            right = self.parse_unary()
            left = Binary(op.value, left, right, op.line)
        return left

    def parse_unary(self) -> Any:
        if self.check_op("-", "!"):
            op = self.advance()
            operand = self.parse_unary()
            return Unary(op.value, operand, op.line)
        return self.parse_primary()

    def parse_primary(self) -> Any:
        tok = self.cur
        if tok.kind == "NUM":
            self.advance()
            return Num(tok.value, tok.line)
        if tok.kind == "STR":
            self.advance()
            return Str(tok.value, tok.line)
        if tok.kind == "KW" and tok.value in ("true", "false"):
            self.advance()
            return Bool(tok.value == "true", tok.line)
        if tok.kind == "OP" and tok.value == "(":
            self.advance()
            expr = self.parse_expression()
            self.expect_op(")", "to close parenthesized expression")
            return expr
        if tok.kind == "IDENT":
            self.advance()
            if self.cur.kind == "OP" and self.cur.value == "(":
                self.advance()
                args: List[Any] = []
                if not (self.cur.kind == "OP" and self.cur.value == ")"):
                    while True:
                        args.append(self.parse_expression())
                        if not self.match_op(","):
                            break
                self.expect_op(")", "to close call argument list")
                return Call(tok.value, args, tok.line)
            return Var(tok.value, tok.line)
        raise ParseError(f"expected expression, found {self._describe(tok)}", tok.line, tok.col)


def parse_source(src: str) -> List[Any]:
    """Tokenize and parse a whole compilation unit into top-level statements."""
    return Parser(tokenize(src)).parse_program(src)


# ------------------------------------------------------------- opcodes -----


class OpCode:
    """Typed opcode identifiers; each Op carries ``code``, ``arg`` and ``line``."""

    CONST_I = "CONST_I"  # int literal
    CONST_F = "CONST_F"  # float literal
    CONST_S = "CONST_S"  # string literal
    CONST_B = "CONST_B"  # bool literal
    LOAD_LOCAL = "LOAD_LOCAL"
    STORE_LOCAL = "STORE_LOCAL"
    LOAD_GLOBAL = "LOAD_GLOBAL"
    STORE_GLOBAL = "STORE_GLOBAL"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    NEG = "NEG"
    NOT = "NOT"
    EQ = "EQ"
    NE = "NE"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    JMP = "JMP"
    JMP_IF_FALSE = "JMP_IF_FALSE"
    JMP_IF_TRUE = "JMP_IF_TRUE"
    CALL = "CALL"
    RET = "RET"
    POP = "POP"
    PRINT = "PRINT"
    HALT = "HALT"

    ALL = frozenset(
        {
            CONST_I, CONST_F, CONST_S, CONST_B,
            LOAD_LOCAL, STORE_LOCAL, LOAD_GLOBAL, STORE_GLOBAL,
            ADD, SUB, MUL, DIV, MOD, NEG, NOT,
            EQ, NE, LT, LE, GT, GE,
            JMP, JMP_IF_FALSE, JMP_IF_TRUE,
            CALL, RET, POP, PRINT, HALT,
        }
    )


@dataclass
class Op:
    code: str
    arg: Any
    line: int


@dataclass
class Program:
    code: List[Op]
    functions: Dict[str, Tuple[int, int, int, int]]  # name -> (entry, n_args, n_locals, line)


# ------------------------------------------------------------- semantics ---


def format_value(value: Any) -> str:
    """Shared print formatting for the VM and the reference evaluator."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "none"
    return str(value)


def truthy(value: Any) -> bool:
    """Python-like truthiness shared by both engines (documented in README)."""
    if value is None or value is False:
        return False
    if value == 0 or value == "" or value == 0.0:
        return False
    return True


def binop(op: str, a: Any, b: Any) -> Any:
    """Apply a binary operator; raise ``ValueError`` with a message on error."""
    if op == "+":
        if isinstance(a, str) and isinstance(b, str):
            return a + b
        if _numeric(a) and _numeric(b):
            return a + b
        raise ValueError(f"invalid operands for '+': {type_name(a)} and {type_name(b)}")
    if op in ("-", "*", "/", "%"):
        if not (_numeric(a) and _numeric(b)):
            raise ValueError(f"invalid operands for '{op}': {type_name(a)} and {type_name(b)}")
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                raise ValueError("division by zero")
            return a / b
        if b == 0:
            raise ValueError("division by zero")
        return a % b
    if op in ("<", "<=", ">", ">="):
        both_nums = _numeric(a) and _numeric(b)
        both_strs = isinstance(a, str) and isinstance(b, str)
        if not (both_nums or both_strs):
            raise ValueError(f"invalid operands for '{op}': {type_name(a)} and {type_name(b)}")
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    raise ValueError(f"unknown operator '{op}'")


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if v is None:
        return "none"
    return type(v).__name__


# ------------------------------------------------------------- compiler ----


class Compiler:
    """Single-pass compiler with jump backpatching and call-site linking."""

    def __init__(self, stmts: List[Any]):
        self.stmts = stmts
        self.code: List[Op] = []
        self.functions: Dict[str, Tuple[int, int, int, int]] = {}
        self.function_defs: List[Function] = []
        self.pending_calls: List[Tuple[int, str, int]] = []  # (ip, name, line)
        self.globals: Dict[str, int] = {}
        self.locals: Dict[str, int] = {}
        self.n_locals = 0
        self.in_function = False

    def emit(self, code: str, arg: Any, line: int) -> int:
        self.code.append(Op(code, arg, line))
        return len(self.code) - 1

    def compile(self) -> Program:
        for stmt in self.stmts:
            self.compile_stmt(stmt)
        self.emit(OpCode.HALT, None, 0)
        for fn in self.function_defs:
            self.compile_function(fn)
        for ip, name, line in self.pending_calls:
            entry = self.functions.get(name)
            if entry is None:
                raise CompileError(f"undefined function '{name}'", line)
            _, n_args, n_locals, _ = entry
            op = self.code[ip]
            want = op.arg[3]
            if want != n_args:
                raise CompileError(
                    f"function '{name}' expects {n_args} arguments, got {want}", line
                )
            self.code[ip] = Op(OpCode.CALL, (entry[0], op.arg[1], n_locals, name), op.line)
        return Program(self.code, self.functions)

    def compile_function(self, fn: Function) -> None:
        if fn.name in self.functions or fn.name in self.globals:
            raise CompileError(f"duplicate definition of '{fn.name}'", fn.line)
        entry = len(self.code)
        saved_locals, saved_n, saved_in = self.locals, self.n_locals, self.in_function
        self.locals = {}
        self.n_locals = 0
        self.in_function = True
        for param in fn.params:
            if param in self.locals:
                raise CompileError(f"duplicate parameter '{param}'", fn.line)
            self.locals[param] = self.n_locals
            self.n_locals += 1
        for stmt in fn.body.stmts:
            self.compile_stmt(stmt)
        self.emit(OpCode.CONST_B, False, fn.line)  # implicit fall-through return
        self.emit(OpCode.RET, None, fn.line)
        self.functions[fn.name] = (entry, len(fn.params), self.n_locals, fn.line)
        self.locals, self.n_locals, self.in_function = saved_locals, saved_n, saved_in

    def declare_local(self, name: str, line: int) -> int:
        if name in self.locals:
            return self.locals[name]  # redeclaration reuses the slot
        slot = self.n_locals
        self.locals[name] = slot
        self.n_locals += 1
        return slot

    def compile_stmt(self, stmt: Any) -> None:
        if isinstance(stmt, Let):
            self.compile_expr(stmt.value)
            if self.in_function:
                slot = self.declare_local(stmt.name, stmt.line)
                self.emit(OpCode.STORE_LOCAL, slot, stmt.line)
                self.emit(OpCode.POP, None, stmt.line)
            else:
                if stmt.name in self.functions:
                    raise CompileError(f"'{stmt.name}' is a function", stmt.line)
                self.globals.setdefault(stmt.name, len(self.globals))
                self.emit(OpCode.STORE_GLOBAL, stmt.name, stmt.line)
                self.emit(OpCode.POP, None, stmt.line)
        elif isinstance(stmt, Assign):
            self.compile_expr(stmt.value)
            if stmt.name in self.locals:
                self.emit(OpCode.STORE_LOCAL, self.locals[stmt.name], stmt.line)
            elif stmt.name in self.globals:
                self.emit(OpCode.STORE_GLOBAL, stmt.name, stmt.line)
            else:
                raise CompileError(f"undefined variable '{stmt.name}'", stmt.line)
            self.emit(OpCode.POP, None, stmt.line)
        elif isinstance(stmt, Print):
            self.compile_expr(stmt.expr)
            self.emit(OpCode.PRINT, None, stmt.line)
        elif isinstance(stmt, If):
            jump_fixups: List[int] = []
            for cond, block in stmt.branches:
                self.compile_expr(cond)
                jf = self.emit(OpCode.JMP_IF_FALSE, None, stmt.line)
                self.compile_block(block)
                jump_fixups.append(self.emit(OpCode.JMP, None, stmt.line))
                # False falls through to the next branch condition (or the
                # else block, which is compiled immediately after this loop).
                self.code[jf] = Op(OpCode.JMP_IF_FALSE, len(self.code), cond.line)
            if stmt.else_block is not None:
                self.compile_block(stmt.else_block)
            for fixup in jump_fixups:
                self.code[fixup] = Op(OpCode.JMP, len(self.code), stmt.line)
        elif isinstance(stmt, While):
            loop_start = len(self.code)
            self.compile_expr(stmt.cond)
            exit_jmp = self.emit(OpCode.JMP_IF_FALSE, None, stmt.line)
            self.compile_block(stmt.body)
            self.emit(OpCode.JMP, loop_start, stmt.line)
            self.code[exit_jmp] = Op(OpCode.JMP_IF_FALSE, len(self.code), stmt.line)
        elif isinstance(stmt, Function):
            self.function_defs.append(stmt)
        elif isinstance(stmt, Return):
            if not self.in_function:
                raise CompileError("'return' outside of a function", stmt.line)
            self.compile_expr(stmt.expr)
            self.emit(OpCode.RET, None, stmt.line)
        elif isinstance(stmt, Block):
            self.compile_block(stmt)
        elif isinstance(stmt, ExprStmt):
            self.compile_expr(stmt.expr)
            self.emit(OpCode.POP, None, stmt.line)
        else:  # pragma: no cover - exhaustive over AST nodes
            raise CompileError(f"cannot compile statement {type(stmt).__name__}", None)

    def compile_block(self, block: Block) -> None:
        for stmt in block.stmts:
            self.compile_stmt(stmt)

    def compile_expr(self, expr: Any) -> None:
        if isinstance(expr, Num):
            code = OpCode.CONST_I if isinstance(expr.value, int) else OpCode.CONST_F
            self.emit(code, expr.value, expr.line)
        elif isinstance(expr, Str):
            self.emit(OpCode.CONST_S, expr.value, expr.line)
        elif isinstance(expr, Bool):
            self.emit(OpCode.CONST_B, expr.value, expr.line)
        elif isinstance(expr, Var):
            if expr.name in self.locals:
                self.emit(OpCode.LOAD_LOCAL, self.locals[expr.name], expr.line)
            elif expr.name in self.globals or expr.name in self.functions:
                self.emit(OpCode.LOAD_GLOBAL, expr.name, expr.line)
            else:
                raise CompileError(f"undefined variable '{expr.name}'", expr.line)
        elif isinstance(expr, Unary):
            self.compile_expr(expr.operand)
            self.emit(OpCode.NEG if expr.op == "-" else OpCode.NOT, None, expr.line)
        elif isinstance(expr, Binary):
            if expr.op == "&&":
                self.compile_expr(expr.left)
                jf = self.emit(OpCode.JMP_IF_FALSE, None, expr.line)
                self.compile_expr(expr.right)
                end = self.emit(OpCode.JMP, None, expr.line)
                self.code[jf] = Op(OpCode.JMP_IF_FALSE, len(self.code), expr.line)
                self.emit(OpCode.CONST_B, False, expr.line)
                self.code[end] = Op(OpCode.JMP, len(self.code), expr.line)
            elif expr.op == "||":
                self.compile_expr(expr.left)
                jt = self.emit(OpCode.JMP_IF_TRUE, None, expr.line)
                self.compile_expr(expr.right)
                end = self.emit(OpCode.JMP, None, expr.line)
                self.code[jt] = Op(OpCode.JMP_IF_TRUE, len(self.code), expr.line)
                self.emit(OpCode.CONST_B, True, expr.line)
                self.code[end] = Op(OpCode.JMP, len(self.code), expr.line)
            else:
                self.compile_expr(expr.left)
                self.compile_expr(expr.right)
                self.emit(
                    {
                        "+": OpCode.ADD, "-": OpCode.SUB, "*": OpCode.MUL,
                        "/": OpCode.DIV, "%": OpCode.MOD,
                        "==": OpCode.EQ, "!=": OpCode.NE,
                        "<": OpCode.LT, "<=": OpCode.LE,
                        ">": OpCode.GT, ">=": OpCode.GE,
                    }[expr.op],
                    None,
                    expr.line,
                )
        elif isinstance(expr, Call):
            if expr.name in self.globals:
                raise CompileError(f"'{expr.name}' is a variable, not a function", expr.line)
            for arg in expr.args:
                self.compile_expr(arg)
            entry = self.functions.get(expr.name)
            arg_info = (
                (entry[0] if entry else -1),
                len(expr.args),
                (entry[2] if entry else 0),
                len(expr.args),  # checked again at link time against real arity
            )
            ip = self.emit(OpCode.CALL, arg_info, expr.line)
            if entry is None:
                self.pending_calls.append((ip, expr.name, expr.line))
            else:
                want = entry[1]
                if want != len(expr.args):
                    raise CompileError(
                        f"function '{expr.name}' expects {want} arguments, got {len(expr.args)}",
                        expr.line,
                    )
        else:  # pragma: no cover - exhaustive over AST nodes
            raise CompileError(f"cannot compile expression {type(expr).__name__}", None)


def compile_source(src: str) -> Program:
    """Parse and compile ``src`` into a :class:`Program`."""
    return Compiler(parse_source(src)).compile()


# ------------------------------------------------------------------ VM -----


@dataclass
class _Frame:
    ret_ip: int
    locals: List[Any]
    stack_base: int


class VM:
    """Stack machine with an instruction budget and tracked stack depth.

    Stack underflow is impossible in well-formed programs because every
    operand-count check runs *before* any pop; a hand-corrupted program is
    rejected with ``VMError`` and leaves the stack untouched.
    """

    def __init__(self, program: Program, stdout: Any = None, budget: int = 1_000_000) -> None:
        self.program = program
        self.stdout = stdout if stdout is not None else sys.stdout
        self.budget = budget
        self.stack: List[Any] = []
        self.frames: List[_Frame] = []
        self.globals: Dict[str, Any] = {}
        self.ip = 0
        self.steps = 0
        self.peak_depth = 0

    def _push(self, value: Any) -> None:
        self.stack.append(value)
        if len(self.stack) > self.peak_depth:
            self.peak_depth = len(self.stack)

    def _need(self, count: int, op: Op) -> None:
        if len(self.stack) < count:
            raise VMError("stack underflow", op.line)

    def run(self) -> None:
        code = self.program.code
        while True:
            if self.ip >= len(code):
                raise VMError("fell off the end of the program", None)
            op = code[self.ip]
            self.steps += 1
            if self.steps > self.budget:
                raise VMError(
                    "instruction budget exceeded (possible infinite loop)", op.line
                )
            self.ip += 1
            c = op.code
            if c == OpCode.CONST_I or c == OpCode.CONST_F or c == OpCode.CONST_S or c == OpCode.CONST_B:
                self._push(op.arg)
            elif c == OpCode.LOAD_LOCAL:
                self._push(self.frames[-1].locals[op.arg])
            elif c == OpCode.STORE_LOCAL:
                self._need(1, op)
                self.frames[-1].locals[op.arg] = self.stack[-1]
            elif c == OpCode.LOAD_GLOBAL:
                if op.arg not in self.globals:
                    raise VMError(f"undefined variable '{op.arg}'", op.line)
                self._push(self.globals[op.arg])
            elif c == OpCode.STORE_GLOBAL:
                self._need(1, op)
                self.globals[op.arg] = self.stack[-1]
            elif c in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD,
                       OpCode.EQ, OpCode.NE, OpCode.LT, OpCode.LE, OpCode.GT, OpCode.GE):
                self._need(2, op)
                b = self.stack.pop()
                a = self.stack.pop()
                try:
                    self._push(binop(_BINOP_NAMES[c], a, b))
                except ValueError as exc:
                    raise VMError(exc.args[0], op.line) from None
            elif c == OpCode.NEG:
                self._need(1, op)
                v = self.stack.pop()
                if not _numeric(v):
                    raise VMError(f"cannot negate {type_name(v)}", op.line)
                self._push(-v)
            elif c == OpCode.NOT:
                self._need(1, op)
                self._push(not truthy(self.stack.pop()))
            elif c in (OpCode.JMP, OpCode.JMP_IF_FALSE, OpCode.JMP_IF_TRUE):
                if c != OpCode.JMP:
                    self._need(1, op)
                    cond = truthy(self.stack.pop())
                    take = (c == OpCode.JMP_IF_FALSE and not cond) or (
                        c == OpCode.JMP_IF_TRUE and cond
                    )
                    if not take:
                        continue
                self.ip = op.arg
            elif c == OpCode.CALL:
                entry, n_args, n_locals, name = op.arg
                self._need(n_args, op)
                args = self.stack[len(self.stack) - n_args:]
                del self.stack[len(self.stack) - n_args:]
                frame = _Frame(ret_ip=self.ip, locals=[None] * n_locals, stack_base=len(self.stack))
                frame.locals[:n_args] = args
                self.frames.append(frame)
                self.ip = entry
            elif c == OpCode.RET:
                if not self.frames:
                    raise VMError("return with no active call frame", op.line)
                self._need(1, op)
                result = self.stack.pop()
                frame = self.frames.pop()
                del self.stack[frame.stack_base:]
                self._push(result)
                self.ip = frame.ret_ip
            elif c == OpCode.POP:
                self._need(1, op)
                self.stack.pop()
            elif c == OpCode.PRINT:
                self._need(1, op)
                self.stdout.write(format_value(self.stack.pop()) + "\n")
            elif c == OpCode.HALT:
                if self.frames:
                    raise VMError("halt with an active call frame", op.line)
                if self.stack:
                    raise VMError("internal: non-empty stack at halt", op.line)
                return
            else:  # pragma: no cover - OpCode.ALL guards opcodes
                raise VMError(f"unknown opcode {c}", op.line)


_BINOP_NAMES = {
    OpCode.ADD: "+", OpCode.SUB: "-", OpCode.MUL: "*", OpCode.DIV: "/", OpCode.MOD: "%",
    OpCode.EQ: "==", OpCode.NE: "!=", OpCode.LT: "<", OpCode.LE: "<=",
    OpCode.GT: ">", OpCode.GE: ">=",
}


# ---------------------------------------------------- reference evaluator ---


class _ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class ReferenceEvaluator:
    """Tree-walking evaluator for the same AST; used to cross-check the VM."""

    def __init__(self, stdout: Any = None) -> None:
        self.stdout = stdout if stdout is not None else sys.stdout
        self.globals: Dict[str, Any] = {}
        self.functions: Dict[str, Function] = {}

    def run(self, stmts: List[Any]) -> None:
        for stmt in stmts:
            if isinstance(stmt, Function):
                if stmt.name in self.functions:
                    raise EvalError(f"duplicate definition of '{stmt.name}'", stmt.line)
                self.functions[stmt.name] = stmt
            else:
                self.exec_stmt(stmt, None)

    def exec_block(self, block: Block, locals_: Optional[Dict[str, Any]]) -> None:
        for stmt in block.stmts:
            self.exec_stmt(stmt, locals_)

    def exec_stmt(self, stmt: Any, locals_: Optional[Dict[str, Any]]) -> None:
        if isinstance(stmt, Let):
            value = self.eval_expr(stmt.value, locals_)
            if stmt.name in self.functions:
                raise EvalError(f"'{stmt.name}' is a function", stmt.line)
            if locals_ is None:
                self.globals[stmt.name] = value
            else:
                locals_[stmt.name] = value
        elif isinstance(stmt, Assign):
            value = self.eval_expr(stmt.value, locals_)
            if locals_ is not None and stmt.name in locals_:
                locals_[stmt.name] = value
            elif stmt.name in self.globals:
                self.globals[stmt.name] = value
            else:
                raise EvalError(f"undefined variable '{stmt.name}'", stmt.line)
        elif isinstance(stmt, Print):
            self.stdout.write(format_value(self.eval_expr(stmt.expr, locals_)) + "\n")
        elif isinstance(stmt, If):
            for cond, block in stmt.branches:
                if truthy(self.eval_expr(cond, locals_)):
                    self.exec_block(block, locals_)
                    return
            if stmt.else_block is not None:
                self.exec_block(stmt.else_block, locals_)
        elif isinstance(stmt, While):
            while truthy(self.eval_expr(stmt.cond, locals_)):
                self.exec_block(stmt.body, locals_)
        elif isinstance(stmt, Function):
            raise EvalError(f"nested function '{stmt.name}' is not allowed", stmt.line)
        elif isinstance(stmt, Return):
            raise _ReturnSignal(self.eval_expr(stmt.expr, locals_))
        elif isinstance(stmt, Block):
            self.exec_block(stmt, locals_)
        elif isinstance(stmt, ExprStmt):
            self.eval_expr(stmt.expr, locals_)
        else:  # pragma: no cover - exhaustive over AST nodes
            raise EvalError(f"cannot execute {type(stmt).__name__}", None)

    def eval_expr(self, expr: Any, locals_: Dict[str, Any]) -> Any:
        if isinstance(expr, Num):
            return expr.value
        if isinstance(expr, Str):
            return expr.value
        if isinstance(expr, Bool):
            return expr.value
        if isinstance(expr, Var):
            if locals_ is not None and expr.name in locals_:
                return locals_[expr.name]
            if expr.name in self.globals:
                return self.globals[expr.name]
            raise EvalError(f"undefined variable '{expr.name}'", expr.line)
        if isinstance(expr, Unary):
            v = self.eval_expr(expr.operand, locals_)
            if expr.op == "-":
                if not _numeric(v):
                    raise EvalError(f"cannot negate {type_name(v)}", expr.line)
                return -v
            return not truthy(v)
        if isinstance(expr, Binary):
            if expr.op == "&&":
                left = self.eval_expr(expr.left, locals_)
                if not truthy(left):
                    return False
                return self.eval_expr(expr.right, locals_)
            if expr.op == "||":
                left = self.eval_expr(expr.left, locals_)
                if truthy(left):
                    return True
                return self.eval_expr(expr.right, locals_)
            a = self.eval_expr(expr.left, locals_)
            b = self.eval_expr(expr.right, locals_)
            try:
                return binop(expr.op, a, b)
            except ValueError as exc:
                raise EvalError(exc.args[0], expr.line) from None
        if isinstance(expr, Call):
            fn = self.functions.get(expr.name)
            if fn is None:
                if expr.name in self.globals:
                    raise EvalError(f"'{expr.name}' is a variable, not a function", expr.line)
                raise EvalError(f"undefined function '{expr.name}'", expr.line)
            if len(expr.args) != len(fn.params):
                raise EvalError(
                    f"function '{expr.name}' expects {len(fn.params)} arguments, got {len(expr.args)}",
                    expr.line,
                )
            call_locals: Dict[str, Any] = {}
            for name, arg in zip(fn.params, expr.args):
                call_locals[name] = self.eval_expr(arg, locals_)
            try:
                self.exec_block(fn.body, call_locals)
            except _ReturnSignal as signal:
                return signal.value
            return False  # implicit fall-through return, same as the VM
        raise EvalError(f"cannot evaluate {type(expr).__name__}", None)  # pragma: no cover


def run_source(src: str, stdout: Any = None, budget: int = 1_000_000) -> str:
    """Compile and execute ``src`` on the VM; returns captured output."""
    import io

    out = stdout if stdout is not None else io.StringIO()
    program = compile_source(src)
    VM(program, stdout=out, budget=budget).run()
    if isinstance(out, io.StringIO):
        return out.getvalue()
    return ""


def reference_run_source(src: str, stdout: Any = None) -> str:
    """Execute ``src`` with the reference evaluator; returns captured output."""
    import io

    out = stdout if stdout is not None else io.StringIO()
    ReferenceEvaluator(stdout=out).run(parse_source(src))
    if isinstance(out, io.StringIO):
        return out.getvalue()
    return ""


# ---------------------------------------------------------- disassembler ---


_OP_COMMENTS = {
    OpCode.CONST_I: "push int literal", OpCode.CONST_F: "push float literal",
    OpCode.CONST_S: "push string literal", OpCode.CONST_B: "push bool literal",
    OpCode.LOAD_LOCAL: "load local", OpCode.STORE_LOCAL: "store local",
    OpCode.LOAD_GLOBAL: "load global", OpCode.STORE_GLOBAL: "store global",
    OpCode.ADD: "add", OpCode.SUB: "subtract", OpCode.MUL: "multiply",
    OpCode.DIV: "divide", OpCode.MOD: "modulo", OpCode.NEG: "negate",
    OpCode.NOT: "logical not", OpCode.EQ: "equal", OpCode.NE: "not equal",
    OpCode.LT: "less than", OpCode.LE: "less or equal", OpCode.GT: "greater than",
    OpCode.GE: "greater or equal", OpCode.JMP: "jump",
    OpCode.JMP_IF_FALSE: "pop; jump if false", OpCode.JMP_IF_TRUE: "pop; jump if true",
    OpCode.CALL: "call", OpCode.RET: "return", OpCode.POP: "discard top",
    OpCode.PRINT: "print", OpCode.HALT: "stop",
}


def disassemble(program: Program) -> str:
    """Render annotated bytecode; stable output used by golden tests."""
    lines: List[str] = []
    for ip, op in enumerate(program.code):
        arg_str = ""
        if op.code in (OpCode.JMP, OpCode.JMP_IF_FALSE, OpCode.JMP_IF_TRUE):
            arg_str = f"{op.arg:04d}"
        elif op.code == OpCode.CALL:
            name = op.arg[3]
            arg_str = f"{name}/{op.arg[1]}@{op.arg[0]:04d}"
        elif op.code in (OpCode.LOAD_LOCAL, OpCode.STORE_LOCAL):
            arg_str = str(op.arg)
        elif op.code in (OpCode.LOAD_GLOBAL, OpCode.STORE_GLOBAL):
            arg_str = str(op.arg)
        elif op.arg is not None:
            arg_str = repr(op.arg) if op.code == OpCode.CONST_S else str(op.arg)
        comment = _OP_COMMENTS.get(op.code, "")
        lines.append(f"{ip:04d}  {op.code:<13} {arg_str:<12}; {comment}".rstrip())
    return "\n".join(lines) + "\n"


def disassemble_source(src: str) -> str:
    """Compile then disassemble ``src``."""
    return disassemble(compile_source(src))


# ------------------------------------------------------------------ CLI ----


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: ``python bytecode_vm.py run file`` / ``python bytecode_vm.py dis file``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in ("run", "dis"):
        print("usage: python bytecode_vm.py (run|dis) <file> [--budget N]")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    mode = argv[0]
    if len(argv) < 2:
        print("error: missing source file", file=sys.stderr)
        return 2
    budget = 1_000_000
    if "--budget" in argv:
        idx = argv.index("--budget")
        try:
            budget = int(argv[idx + 1])
        except (IndexError, ValueError):
            print("error: --budget requires an integer", file=sys.stderr)
            return 2
    try:
        with open(argv[1], "r", encoding="utf-8") as handle:
            src = handle.read()
    except OSError as exc:
        print(f"error: cannot read {argv[1]}: {exc}", file=sys.stderr)
        return 2
    try:
        if mode == "dis":
            sys.stdout.write(disassemble_source(src))
        else:
            run_source(src, stdout=sys.stdout, budget=budget)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

