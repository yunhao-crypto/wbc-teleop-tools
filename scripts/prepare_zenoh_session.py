#!/usr/bin/env python3
"""Make a private session config without outgoing automatic SHM allocation.

The vendor file (including endpoints/authentication) is copied, never edited.
This only changes transport.shared_memory.transport_optimization.enabled.
The generated file can contain credentials: mode 0600, never include in Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def patch_config(source: str) -> str:
    # Tokenize strings/comments too, so a comment or password cannot masquerade
    # as a configuration key. Retain original byte positions for the one edit.
    lexer = re.compile(
        r'\s+|//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"'
        r"|'(?:\\.|[^'\\])*'|[{}\[\]:,]|[^\s{}\[\]:,]+"
    )
    tokens = [m for m in lexer.finditer(source)
              if not m.group().isspace() and not m.group().startswith(('//', '/*'))]
    matches = []

    def value(index, path):
        token = tokens[index].group()
        if token == '{':
            index += 1
            while tokens[index].group() != '}':
                key = tokens[index].group().strip('"\'')
                if tokens[index + 1].group() != ':':
                    raise ValueError('Expected configuration key/value')
                index = value(index + 2, (*path, key))
                if tokens[index].group() == ',':
                    index += 1
                elif tokens[index].group() != '}':
                    raise ValueError('Expected configuration separator')
            return index + 1
        if token == '[':
            index += 1
            while tokens[index].group() != ']':
                index = value(index, (*path, '[]'))
                if tokens[index].group() == ',':
                    index += 1
                elif tokens[index].group() != ']':
                    raise ValueError('Expected array separator')
            return index + 1
        if path == ('transport', 'shared_memory', 'transport_optimization', 'enabled'):
            if token not in ('true', 'false'):
                raise ValueError('Automatic SHM setting must be boolean')
            matches.append(tokens[index])
        return index + 1

    try:
        consumed = value(0, ())
    except IndexError as exc:
        raise ValueError('Incomplete session configuration') from exc
    if consumed != len(tokens) or len(matches) != 1:
        raise ValueError('Require exactly one outgoing automatic SHM setting')
    match = matches[0]
    return source[:match.start()] + 'false' + source[match.end():]


def prepare(source: Path, output: Path):
    if source.resolve() == output.resolve():
        raise ValueError('Cannot overwrite vendor session configuration')
    original = source.read_text()
    patched = patch_config(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=output.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            os.chmod(temporary, 0o600)
            f.write(patched)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = dict(source=str(source), output=str(output),
                   source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   output_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                   changed_setting='transport.shared_memory.transport_optimization.enabled',
                   value=False)
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
